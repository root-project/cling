//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/ValuePrinter.h"

#include "cling/Interpreter/CValuePrinter.h"
#include "cling/Interpreter/ValuePrinterInfo.h"
#include "cling/Interpreter/StoredValueRef.h"
#include "cling/Interpreter/Value.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/ExecutionEngine/GenericValue.h"

#include <string>
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
extern "C" void cling_PrintValue(void* /*clang::Expr**/ E,
                      void* /*clang::ASTContext**/ C,
                      const void* value) {
  clang::Expr* Exp = (clang::Expr*)E;
  clang::ASTContext* Context = (clang::ASTContext*)C;
  ValuePrinterInfo VPI(Exp->getType(), Context);

  // We need stream that doesn't close its file descriptor, thus we are not
  // using llvm::outs. Keeping file descriptor open we will be able to use
  // the results in pipes (Savannah #99234).
  llvm::raw_fd_ostream outs (STDOUT_FILENO, /*ShouldClose*/false);

  valuePrinterInternal::flushToStream(outs, printType(value, value, VPI)
                                      + printValue(value, value, VPI));
}


static void StreamValue(llvm::raw_ostream& o, const void* const p,
                        const ValuePrinterInfo& VPI);

static void StreamChar(llvm::raw_ostream& o, const char v) {
  o << '"' << v << "\"";
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

static void StreamRef(llvm::raw_ostream& o, const void* v) {
  o <<"&" << v;
}

static void StreamPtr(llvm::raw_ostream& o, const void* v) {
  o << v;
}

static void StreamArr(llvm::raw_ostream& o, const void* p,
                      const ValuePrinterInfo& VPI) {
  const clang::QualType& Ty = VPI.getType();
  clang::ASTContext& C = *VPI.getASTContext();
  const clang::ArrayType* ArrTy = Ty->getAsArrayTypeUnsafe();
  clang::QualType ElementTy = ArrTy->getElementType();
  if (ElementTy->isCharType())
    StreamCharPtr(o, (const char*)p);
  else if (Ty->isConstantArrayType()) {
    // Stream a constant array by streaming up to 5 elements.
    const clang::ConstantArrayType* CArrTy
      = C.getAsConstantArrayType(Ty);
    const llvm::APInt& APSize = CArrTy->getSize();
    size_t ElBytes = C.getTypeSize(ElementTy) / C.getCharWidth();
    size_t Size = (size_t)APSize.getZExtValue();
    o << "{ ";
    ValuePrinterInfo ElVPI(ElementTy, &C);
    for (size_t i = 0; i < Size; ++i) {
      StreamValue(o, ((const char*)p) + i * ElBytes, ElVPI);
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
    StreamPtr(o, p);
}

static void StreamFunction(llvm::raw_ostream& o, const void* addr,
                           ValuePrinterInfo VPI) {
  o << "Function @" << addr << '\n';

  const clang::DeclRefExpr* DeclRefExp
    = llvm::dyn_cast_or_null<clang::DeclRefExpr>(VPI.getExpr());
  const clang::FunctionDecl* FD = 0;
  if (DeclRefExp)
    FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(DeclRefExp->getDecl());
  if (FD) {
    clang::SourceRange SRange = FD->getSourceRange();
    const char* cBegin = 0;
    const char* cEnd = 0;
    bool Invalid;
    if (SRange.isValid()) {
      clang::SourceManager& SM = VPI.getASTContext()->getSourceManager();
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
  } else {
    o << ":\n";
    // type-based printing:
    VPI.getType().print(o, VPI.getASTContext()->getPrintingPolicy());
  }
  // type-based print() never and decl-based print() sometimes does not include
  // a final newline:
  o << '\n';
}

static void StreamClingValue(llvm::raw_ostream& o, const Value* value,
                             clang::ASTContext& C) {
  if (!value || !value->isValid()) {
    o << "<<<invalid>>> @" << value;
  } else {
    o << "boxes [";
    o << "("
      << value->getClangType().getAsString(C.getPrintingPolicy())
      << ") ";
    clang::QualType valType = value->getClangType().getDesugaredType(C);
    if (valType->isFloatingType())
      o << value->getGV().DoubleVal;
    else if (valType->isIntegerType())
      o << value->getGV().IntVal.getSExtValue();
    else if (valType->isBooleanType())
      o << value->getGV().IntVal.getBoolValue();
    else
      StreamValue(o, value->getGV().PointerVal,
                  ValuePrinterInfo(valType, &C));
    o << "]";
  }
}

static void StreamObj(llvm::raw_ostream& o, const void* v,
                      const ValuePrinterInfo& VPI) {
  const clang::Type* Ty = VPI.getType().getTypePtr();
  if (clang::CXXRecordDecl* CXXRD = Ty->getAsCXXRecordDecl()) {
    std::string QualName = CXXRD->getQualifiedNameAsString();
    if (QualName == "cling::StoredValueRef"){
      valuePrinterInternal::StreamStoredValueRef(o, (const StoredValueRef*)v,
                                                 *VPI.getASTContext());
      return;
    } else if (QualName == "cling::Value") {
      StreamClingValue(o, (const Value*)v, *VPI.getASTContext());
      return;
    }
  } // if CXXRecordDecl

  // TODO: Print the object members.
  o << "@" << v;
}

static void StreamValue(llvm::raw_ostream& o, const void* const p,
                        const ValuePrinterInfo& VPI) {
  clang::ASTContext& C = *VPI.getASTContext();
  clang::QualType Ty = VPI.getType().getDesugaredType(C);
  if (const clang::BuiltinType *BT
           = llvm::dyn_cast<clang::BuiltinType>(Ty.getCanonicalType())) {
    switch (BT->getKind()) {
    case clang::BuiltinType::Bool:
      if (*(const bool*)p) o << "true";
      else o << "false"; break;
    case clang::BuiltinType::Char_U:
    case clang::BuiltinType::UChar:
    case clang::BuiltinType::Char_S:
    case clang::BuiltinType::SChar:  StreamChar(o, *(const char*)p); break;
    case clang::BuiltinType::Short:  o << *(const short*)p; break;
    case clang::BuiltinType::UShort:
      o << *(const unsigned short*)p;
      break;
    case clang::BuiltinType::Int:    o << *(const int*)p; break;
    case clang::BuiltinType::UInt:
      o << *(const unsigned int*)p;
      break;
    case clang::BuiltinType::Long:   o << *(const long*)p; break;
    case clang::BuiltinType::ULong:
      o << *(const unsigned long*)p;
      break;
    case clang::BuiltinType::LongLong:
      o << *(const long long*)p;
      break;
    case clang::BuiltinType::ULongLong:
      o << *(const unsigned long long*)p;
      break;
    case clang::BuiltinType::Float:  o << *(const float*)p; break;
    case clang::BuiltinType::Double: o << *(const double*)p; break;
    default:
      StreamObj(o, p, ValuePrinterInfo(Ty, &C));
    }
  }
  else if (Ty.getAsString().compare("class std::basic_string<char>") == 0) {
    StreamObj(o, p, ValuePrinterInfo(Ty, &C));
    o << " "; // force a space
    o <<"c_str: ";
    StreamCharPtr(o, ((const char*) (*(const std::string*)p).c_str()));
  }
  else if (Ty->isEnumeralType()) {
    clang::EnumDecl* ED = Ty->getAs<clang::EnumType>()->getDecl();
    uint64_t value = *(const uint64_t*)p;
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
    StreamRef(o, p);
  else if (Ty->isPointerType()) {
    clang::QualType PointeeTy = Ty->getPointeeType();
    if (PointeeTy->isCharType())
      StreamCharPtr(o, (const char*)p);
    else
      StreamPtr(o, p);
  }
  else if (Ty->isArrayType())
    StreamArr(o, p, ValuePrinterInfo(Ty, &C));
  else if (Ty->isFunctionType())
     StreamFunction(o, p, VPI);
  else
    StreamObj(o, p, ValuePrinterInfo(Ty, &C));
}

namespace cling {
namespace valuePrinterInternal {
  std::string printValue_Default(const void* const p,
                                 const ValuePrinterInfo& VPI) {
    std::string buf;
    {
      llvm::raw_string_ostream o(buf);
      StreamValue(o, p, VPI);
    }
    return buf;
  }

  std::string printType_Default(const ValuePrinterInfo& VPI) {
    std::string buf;
    {
      llvm::raw_string_ostream o(buf);
      o << "(";
      o << VPI.getType().getAsString();
      o << ") ";
    }
    return buf;
  }

  void StreamStoredValueRef(llvm::raw_ostream& o,
                            const StoredValueRef* VR,
                            clang::ASTContext& C) {
    if (VR->isValid()) {
      StreamClingValue(o, &VR->get(), C);
    } else {
      o << "<<<invalid>>> @" << VR;
    }
  }

  void flushToStream(llvm::raw_ostream& o, const std::string& s) {
    // We want to keep stdout and o in sync if o is different from stdout.
    fflush(stdout);
    o << s;
    o.flush();
  }
} // end namespace valuePrinterInternal
} // end namespace cling
