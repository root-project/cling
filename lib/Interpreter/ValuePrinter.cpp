//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Value.h"

#include "cling/Interpreter/CValuePrinter.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Interpreter/Value.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/ExecutionEngine/GenericValue.h"

#include <string>
#include <sstream>
#include <cstdio>

// Fragment copied from LLVM's raw_ostream.cpp
#if defined(_MSC_VER)
#ifndef STDIN_FILENO
# define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
# define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
# define STDERR_FILENO 2
#endif
#else
//#if defined(HAVE_UNISTD_H)
# include <unistd.h>
//#endif
#endif

using namespace cling;

// Implements the CValuePrinter interface.
extern "C" void cling_PrintValue(void* /*cling::Value**/ V) {
  //Value* value = (Value*)V;

  // We need stream that doesn't close its file descriptor, thus we are not
  // using llvm::outs. Keeping file descriptor open we will be able to use
  // the results in pipes (Savannah #99234).
  //llvm::raw_fd_ostream outs (STDOUT_FILENO, /*ShouldClose*/false);

  //std::string typeStr = printType(value->getPtr(), value->getPtr(), *value);
  //std::string valueStr = printValue(value->getPtr(), value->getPtr(), *value);
}


static void StreamValue(llvm::raw_ostream& o, const void* V, clang::QualType QT,
                        cling::Interpreter& interp);

static void StreamChar(llvm::raw_ostream& o, const char v) {
  if (isprint(v))
    o << '\'' << v << '\'';
  else {
    o << "\\0x";
    o.write_hex(v);
  }
}

static void StreamCharPtr(llvm::raw_ostream& o, const char* const v) {
  if (!v) {
    o << "<<<NULL>>>";
    return;
  }
  o << '"';
  const char* p = v;
  for (;*p && p - v < 128; ++p) {
    o << *p;
  }
  if (*p) o << "\"...";
  else o << "\"";
}

static void StreamRef(llvm::raw_ostream& o, const void** V, clang::QualType Ty,
                       cling::Interpreter& interp){
  const clang::ReferenceType* RTy = llvm::dyn_cast<clang::ReferenceType>(Ty);
  StreamValue(o, *V, RTy->getPointeeTypeAsWritten(), interp);
}

static void StreamPtr(llvm::raw_ostream& o, const void* v) {
  o << v;
}

static void StreamArr(llvm::raw_ostream& o, const void* V, clang::QualType Ty,
                      cling::Interpreter& interp) {
  clang::ASTContext& C = interp.getCI()->getASTContext();
  const clang::ArrayType* ArrTy = Ty->getAsArrayTypeUnsafe();
  clang::QualType ElementTy = ArrTy->getElementType();
  if (ElementTy->isCharType())
    StreamCharPtr(o, (const char*)V);
  else if (Ty->isConstantArrayType()) {
    // Stream a constant array by streaming up to 5 elements.
    const clang::ConstantArrayType* CArrTy
      = C.getAsConstantArrayType(Ty);
    const llvm::APInt& APSize = CArrTy->getSize();
    size_t ElBytes = C.getTypeSize(ElementTy) / C.getCharWidth();
    size_t Size = (size_t)APSize.getZExtValue();
    o << "{ ";
    for (size_t i = 0; i < Size; ++i) {
      // Handle the case of constant size array of pointers. Eg. const char*[]
      if (ElementTy->isPointerType())
        StreamValue(o, *(const char* const *)V + i * ElBytes, ElementTy, interp);
      else
        StreamValue(o, (const char*)V + i * ElBytes, ElementTy, interp);

      if (i + 1 < Size) {
        if (i == 4) {
          o << "...";
          break;
        }
        else o << ", ";
      }
    }
    o << " }";
  } else
    StreamPtr(o, V);
}

static void StreamFunction(llvm::raw_ostream& o, const void* V,
                           clang::QualType QT, cling::Interpreter& Interp) {
  o << "Function @" << V << '\n';

  clang::ASTContext& C = Interp.getCI()->getASTContext();
  const Transaction* T = Interp.getLastTransaction();
  assert(T->getWrapperFD() && "Must have a wrapper.");
  clang::FunctionDecl* WrapperFD = T->getWrapperFD();

  const clang::FunctionDecl* FD = 0;
  // CE should be the setValueNoAlloc call expr.
  if (const clang::CallExpr* CallE
    = llvm::dyn_cast_or_null<clang::CallExpr>(
                                  utils::Analyze::GetOrCreateLastExpr(WrapperFD,
                                                                /*foundAtPos*/0,
                                                                /*omitDS*/false,
                                                          &Interp.getSema()))) {
    if (const clang::FunctionDecl* FDsetValue
        = llvm::dyn_cast_or_null<clang::FunctionDecl>(CallE->getCalleeDecl())){
      if (FDsetValue->getNameAsString() == "setValueNoAlloc" &&
          CallE->getNumArgs() == 5) {
        const clang::Expr* Arg4 = CallE->getArg(4);
        while (const clang::CastExpr* CastE
               = clang::dyn_cast<clang::CastExpr>(Arg4))
          Arg4 = CastE->getSubExpr();
        if (const clang::DeclRefExpr* DeclRefExp
            = llvm::dyn_cast<clang::DeclRefExpr>(Arg4))
          FD = llvm::dyn_cast<clang::FunctionDecl>(DeclRefExp->getDecl());
      }
    }
  }

  if (FD) {
    clang::SourceRange SRange = FD->getSourceRange();
    const char* cBegin = 0;
    const char* cEnd = 0;
    bool Invalid;
    if (SRange.isValid()) {
      clang::SourceManager& SM = C.getSourceManager();
      clang::SourceLocation LocBegin = SRange.getBegin();
      LocBegin = SM.getExpansionRange(LocBegin).first;
      o << "  at " << SM.getFilename(LocBegin);
      unsigned LineNo = SM.getSpellingLineNumber(LocBegin, &Invalid);
      if (!Invalid)
        o << ':' << LineNo;
      o << ":\n";
      bool Invalid = false;
      cBegin = SM.getCharacterData(LocBegin, &Invalid);
      if (!Invalid) {
        clang::SourceLocation LocEnd = SRange.getEnd();
        LocEnd = SM.getExpansionRange(LocEnd).second;
        cEnd = SM.getCharacterData(LocEnd, &Invalid);
        if (Invalid)
          cBegin = 0;
      } else {
        cBegin = 0;
      }
    }
    if (cBegin && cEnd && cEnd > cBegin && cEnd - cBegin < 16 * 1024) {
      o << llvm::StringRef(cBegin, cEnd - cBegin + 1);
    } else {
      const clang::FunctionDecl* FDef;
      if (FD->hasBody(FDef))
        FD = FDef;
      FD->print(o);
      //const clang::FunctionDecl* FD
      //  = llvm::cast<const clang::FunctionType>(Ty)->getDecl();
    }
  }
  // type-based print() never and decl-based print() sometimes does not include
  // a final newline:
  o << '\n';
}

static void StreamLongDouble(llvm::raw_ostream& o, const Value* value,
                             clang::ASTContext& C) {
  std::stringstream sstr;
  sstr << value->simplisticCastAs<long double>();
  o << sstr.str() << 'L';
}

static void StreamClingValue(llvm::raw_ostream& o, const Value* value) {
  if (!value || !value->isValid()) {
    o << "<<<invalid>>> @" << value;
  } else {
    clang::ASTContext& C = value->getASTContext();
    clang::QualType QT = value->getType();
    o << "boxes [";
    o << "("
      << QT.getAsString(C.getPrintingPolicy())
      << ") ";
    clang::QualType valType = QT.getDesugaredType(C).getNonReferenceType();
    if (C.hasSameType(valType, C.LongDoubleTy))
      StreamLongDouble(o, value, C);
    else if (valType->isFloatingType())
      o << value->simplisticCastAs<double>();
    else if (valType->isIntegerType()) {
      if (valType->hasSignedIntegerRepresentation())
        o << value->simplisticCastAs<long long>();
      else
        o << value->simplisticCastAs<unsigned long long>();
    } else if (valType->isBooleanType())
      o << (value->simplisticCastAs<bool>() ? "true" : "false");
    else if (!valType->isVoidType())
      StreamValue(o, value->getPtr(), valType,
                  *const_cast<Interpreter*>(value->getInterpreter()));
    o << "]";
  }
}

static void StreamObj(llvm::raw_ostream& o, const void* V, clang::QualType Ty) {
  if (clang::CXXRecordDecl* CXXRD = Ty->getAsCXXRecordDecl()) {
    std::string QualName = CXXRD->getQualifiedNameAsString();
    if (QualName == "cling::Value"){
      StreamClingValue(o, (const cling::Value*)V);
      return;
    }
  } // if CXXRecordDecl

  // TODO: Print the object members.
  o << "@" << V;
}

static void StreamValue(llvm::raw_ostream& o, const void* V,
                        clang::QualType Ty, cling::Interpreter& Interp) {
  clang::ASTContext& C = Interp.getCI()->getASTContext();
  if (const clang::BuiltinType *BT
           = llvm::dyn_cast<clang::BuiltinType>(Ty.getCanonicalType())) {
    switch (BT->getKind()) {
    case clang::BuiltinType::Bool:
      if (*(const bool*)V)
        o << "true";
      else
        o << "false";
      break;
    case clang::BuiltinType::Char_U: // intentional fall through
    case clang::BuiltinType::UChar: // intentional fall through
    case clang::BuiltinType::Char_S: // intentional fall through
    case clang::BuiltinType::SChar:
      StreamChar(o, *(const char*)V); break;
    case clang::BuiltinType::Short:
      o << *(const short*)V; break;
    case clang::BuiltinType::UShort:
      o << *(const unsigned short*)V; break;
    case clang::BuiltinType::Int:
      o << *(const int*)V; break;
    case clang::BuiltinType::UInt:
      o << *(const unsigned int*)V; break;
    case clang::BuiltinType::Long:
      o << *(const long*)V; break;
    case clang::BuiltinType::ULong:
      o << *(const unsigned long*)V; break;
    case clang::BuiltinType::LongLong:
      o << *(const long long*)V; break;
    case clang::BuiltinType::ULongLong:
      o << *(const unsigned long long*)V; break;
    case clang::BuiltinType::Float:
      o << *(const float*)V; break;
    case clang::BuiltinType::Double:
      o << *(const double*)V; break;
    case clang::BuiltinType::LongDouble: {
      std::stringstream ssLD;
      ssLD << *(const long double*)V;
      o << ssLD.str() << 'L'; break;
    }
    default:
      StreamObj(o, V, Ty);
    }
  }
  else if (Ty.getAsString().compare("std::string") == 0) {
    StreamObj(o, V, Ty);
    o << " "; // force a space
    o <<"c_str: ";
    StreamCharPtr(o, ((const char*) (*(const std::string*)V).c_str()));
  }
  else if (Ty->isEnumeralType()) {
    clang::EnumDecl* ED = Ty->getAs<clang::EnumType>()->getDecl();
    uint64_t value = *(const uint64_t*)V;
    bool IsFirst = true;
    llvm::APSInt ValAsAPSInt = C.MakeIntValue(value, Ty);
    for (clang::EnumDecl::enumerator_iterator I = ED->enumerator_begin(),
           E = ED->enumerator_end(); I != E; ++I) {
      if (I->getInitVal() == ValAsAPSInt) {
        if (!IsFirst) {
          o << " ? ";
        }
        o << "(" << I->getQualifiedNameAsString() << ")";
        IsFirst = false;
      }
    }
    o << " : (int) " << ValAsAPSInt.toString(/*Radix = */10);
  }
  else if (Ty->isReferenceType())
    StreamRef(o, (const void**)&V, Ty, Interp);
  else if (Ty->isPointerType()) {
    clang::QualType PointeeTy = Ty->getPointeeType();
    if (PointeeTy->isCharType())
      StreamCharPtr(o, (const char*)V);
    else if (PointeeTy->isFunctionProtoType())
      StreamFunction(o, V, PointeeTy, Interp);
    else
      StreamPtr(o, V);
  }
  else if (Ty->isArrayType())
    StreamArr(o, V, Ty, Interp);
  else if (Ty->isFunctionType())
    StreamFunction(o, V, Ty, Interp);
  else
    StreamObj(o, V, Ty);
}

namespace cling {
namespace valuePrinterInternal {
  void printValue_Default(llvm::raw_ostream& o, const Value& V) {
    clang::ASTContext& C = V.getASTContext();
    clang::QualType Ty = V.getType().getDesugaredType(C);
    Interpreter& Interp = *const_cast<Interpreter*>(V.getInterpreter());
    if (const clang::BuiltinType *BT
        = llvm::dyn_cast<clang::BuiltinType>(Ty.getCanonicalType())) {
      switch (BT->getKind()) {
      case clang::BuiltinType::Bool: // intentional fall through
      case clang::BuiltinType::Char_U: // intentional fall through
      case clang::BuiltinType::Char_S: // intentional fall through
      case clang::BuiltinType::SChar: // intentional fall through
      case clang::BuiltinType::Short: // intentional fall through
      case clang::BuiltinType::Int: // intentional fall through
      case clang::BuiltinType::Long: // intentional fall through
      case clang::BuiltinType::LongLong: {
        long long res = V.getLL();
        StreamValue(o, (const void*)&res, Ty, Interp);
      }
        break;
      case clang::BuiltinType::UChar: // intentional fall through
      case clang::BuiltinType::UShort: // intentional fall through
      case clang::BuiltinType::UInt: // intentional fall through
      case clang::BuiltinType::ULong: // intentional fall through
      case clang::BuiltinType::ULongLong: {
        unsigned long long res = V.getULL();
        StreamValue(o, (const void*)&res, Ty, Interp);
      }
        break;
      case clang::BuiltinType::Float: {
        float res = V.getFloat();
        StreamValue(o, (const void*)&res, Ty, Interp);
      }
        break;
      case clang::BuiltinType::Double: {
        double res = V.getDouble();
        StreamValue(o, (const void*)&res, Ty, Interp);
      }
        break;
      case clang::BuiltinType::LongDouble: {
        long double res = V.getLongDouble();
        StreamValue(o, (const void*)&res, Ty, Interp);
      }
        break;
      default:
        StreamValue(o, V.getPtr(), Ty, Interp);
        break;
      }
    }
    else if (Ty->isIntegralOrEnumerationType()) {
      long long res = V.getLL();
      StreamValue(o, &res, Ty, Interp);
    }
    else if (Ty->isFunctionType())
      StreamValue(o, &V, Ty, Interp);
    else if (Ty->isPointerType() || Ty->isReferenceType()
              || Ty->isArrayType())
      StreamValue(o, V.getPtr(), Ty, Interp);
    else {
      // struct case.
      StreamValue(o, V.getPtr(), Ty, Interp);
    }
  }

  void printType_Default(llvm::raw_ostream& o, const Value& V) {
    using namespace clang;
    QualType QT = V.getType().getNonReferenceType();
    std::string ValueTyStr;
    if (const TypedefType* TDTy = dyn_cast<TypedefType>(QT))
      ValueTyStr = TDTy->getDecl()->getQualifiedNameAsString();
    else if (const TagType* TTy = dyn_cast<TagType>(QT))
      ValueTyStr = TTy->getDecl()->getQualifiedNameAsString();

    if (ValueTyStr.empty())
      ValueTyStr = QT.getAsString();
    else if (QT.hasQualifiers())
      ValueTyStr = QT.getQualifiers().getAsString() + " " + ValueTyStr;

    o << "(";
    o << ValueTyStr;
    if (V.getType()->isReferenceType())
      o << " &";
    o << ") ";
  }
} // end namespace valuePrinterInternal
} // end namespace cling
