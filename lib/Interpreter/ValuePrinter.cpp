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
#include "cling/Utils/Validation.h"

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
#if defined(LLVM_ON_WIN32)
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

// For address validation
#ifdef LLVM_ON_WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

using namespace cling;

// Implements the CValuePrinter interface.
extern "C" void cling_PrintValue(void * /*cling::Value**/ V) {
  //Value* value = (Value*)V;

  // We need stream that doesn't close its file descriptor, thus we are not
  // using llvm::outs. Keeping file descriptor open we will be able to use
  // the results in pipes (Savannah #99234).
  //llvm::raw_fd_ostream outs (STDOUT_FILENO, /*ShouldClose*/false);

  //std::string typeStr = printTypeInternal(*value);
  //std::string valueStr = printValueInternal(*value);
}

static std::string getTypeString(const Value &V) {
  std::ostringstream strm;
  clang::ASTContext &C = V.getASTContext();
  clang::QualType Ty = V.getType().getDesugaredType(C).getNonReferenceType();
  std::string type = cling::utils::TypeName::GetFullyQualifiedName(Ty, C);
  std::ostringstream typeWithOptDeref;

  if (llvm::dyn_cast<clang::BuiltinType>(Ty.getCanonicalType())) {
    typeWithOptDeref << "(" << type << "*)";
  } else if (Ty->isPointerType()) {
    if (Ty->getPointeeType()->isCharType()) {
      // Print char pointers as strings.
      typeWithOptDeref << "(" << type << "*)";
    } else {
      // Fallback to void pointer for other pointers and print the address.
      typeWithOptDeref << "(const void**)";
    }
  }
  else if (Ty->isArrayType()) {
    const clang::ArrayType *ArrTy = Ty->getAsArrayTypeUnsafe();
    clang::QualType ElementTy = ArrTy->getElementType();
    // In case of char ElementTy, printing as string
    if (ElementTy->isCharType()) {
      typeWithOptDeref << "(const char **)";
    } else if (Ty->isConstantArrayType()) {
      const clang::ConstantArrayType *CArrTy = C.getAsConstantArrayType(Ty);
      const llvm::APInt &APSize = CArrTy->getSize();
      size_t size = (size_t) APSize.getZExtValue();

      // typeWithOptDeref example for int[40] array: "((int(*)[40])*(void**)0x5c8f260)"
      typeWithOptDeref << "(" << cling::utils::TypeName::GetFullyQualifiedName(ElementTy, C) << "(*)[" << size << "])*(void**)";
    } else {
      typeWithOptDeref << "(void**)";
    }
  }
  else {
    // In other cases, dereference the address of the object.
    // If no overload or specific template matches,
    // the general template will be used which only prints the address.
    typeWithOptDeref << "*(" << type << "**)";
  }

  strm << typeWithOptDeref.str();
  return strm.str();
}

namespace {
/// RAII object to disable and then re-enable access control in the LangOptions.
struct AccessCtrlRAII_t {
  bool savedAccessControl;
  clang::LangOptions& LangOpts;

  AccessCtrlRAII_t(cling::Interpreter& Interp):
    LangOpts(const_cast<clang::LangOptions&>(Interp.getCI()->getLangOpts())) {
    savedAccessControl = LangOpts.AccessControl;
  }

  ~AccessCtrlRAII_t() {
    LangOpts.AccessControl = savedAccessControl;
  }

};

#ifndef NDEBUG
/// Is typenam parsable?
bool canParseTypeName(cling::Interpreter& Interp, llvm::StringRef typenam) {

  AccessCtrlRAII_t AccessCtrlRAII(Interp);

  cling::Interpreter::CompilationResult Res
    = Interp.declare("namespace { void* cling_printValue_Failure_Typename_check"
                     " = (void*)" + typenam.str() + "nullptr; }");
  if (Res != cling::Interpreter::kSuccess)
    llvm::errs() << "ERROR in cling::executePrintValue(): "
                      "this typename cannot be spelled.\n";
  return Res == cling::Interpreter::kSuccess;
}
#endif

static std::string printQualType(clang::ASTContext& Ctx, clang::QualType QT) {
  using namespace clang;
  std::ostringstream strm;
  QualType QTNonRef = QT.getNonReferenceType();
  std::string ValueTyStr;
  if (const TypedefType *TDTy = dyn_cast<TypedefType>(QTNonRef))
    ValueTyStr = TDTy->getDecl()->getQualifiedNameAsString();
  else if (const TagType *TTy = dyn_cast<TagType>(QTNonRef))
    ValueTyStr = TTy->getDecl()->getQualifiedNameAsString();

  if (ValueTyStr.empty())
    ValueTyStr = cling::utils::TypeName::GetFullyQualifiedName(QTNonRef, Ctx);
  else if (QTNonRef.hasQualifiers())
    ValueTyStr = QTNonRef.getQualifiers().getAsString() + " " + ValueTyStr;

  strm << "(";
  strm << ValueTyStr;
  if (QT->isReferenceType())
    strm << " &";
  strm << ")";
  return strm.str();
}

} // anonymous namespace

template<typename T>
static std::string executePrintValue(const Value &V, const T &val) {
  // don't use std::stringstream, since it doesn't prepend '0x'
  // in front of hexadecimal values when streaming pointer values
  std::string strval;
  llvm::raw_string_ostream printValueSS(strval);
  printValueSS << "cling::printValue(";
  printValueSS << getTypeString(V);
  printValueSS << (const void *) &val;
  printValueSS << ");";

  Interpreter *Interp = V.getInterpreter();
  Value printValueV;

  {
    // We really don't care about protected types here (ROOT-7426)
    AccessCtrlRAII_t AccessCtrlRAII(*Interp);
    clang::DiagnosticsEngine& Diag = Interp->getCI()->getDiagnostics();
    bool oldSuppDiags = Diag.getSuppressAllDiagnostics();
    Diag.setSuppressAllDiagnostics(true);
    Interp->evaluate(printValueSS.str(), printValueV);
    Diag.setSuppressAllDiagnostics(oldSuppDiags);
  }

  if (!printValueV.isValid() || printValueV.getPtr() == nullptr) {
    // That didn't work. We probably diagnosed the issue as part of evaluate().
    llvm::errs() << "ERROR in cling::executePrintValue(): cannot pass value!\n";

    // Check that the issue comes from an unparsable type name: lambdas, unnamed
    // namespaces, types declared inside functions etc. Assert on everything
    // else.
    assert(!canParseTypeName(*Interp, getTypeString(V))
           && "printValue failed on a valid type name.");

    return "ERROR in cling::executePrintValue(): missing value string.";
  }

  return *(std::string *) printValueV.getPtr();
}

static std::string invokePrintValueOverload(const Value &V) {
  clang::ASTContext &C = V.getASTContext();
  clang::QualType Ty = V.getType().getDesugaredType(C).getCanonicalType();
  if (const clang::BuiltinType *BT
      = llvm::dyn_cast<clang::BuiltinType>(Ty.getTypePtr())) {
    switch (BT->getKind()) {
      case clang::BuiltinType::Bool:
        return executePrintValue<bool>(V, V.getLL());

      case clang::BuiltinType::Char_S:
        return executePrintValue<signed char>(V, V.getLL());
      case clang::BuiltinType::SChar:
        return executePrintValue<signed char>(V, V.getLL());
      case clang::BuiltinType::Short:
        return executePrintValue<short>(V, V.getLL());
      case clang::BuiltinType::Int:
        return executePrintValue<int>(V, V.getLL());
      case clang::BuiltinType::Long:
        return executePrintValue<long>(V, V.getLL());
      case clang::BuiltinType::LongLong:
        return executePrintValue<long long>(V, V.getLL());

      case clang::BuiltinType::Char_U:
        return executePrintValue<unsigned char>(V, V.getULL());
      case clang::BuiltinType::UChar:
        return executePrintValue<unsigned char>(V, V.getULL());
      case clang::BuiltinType::UShort:
        return executePrintValue<unsigned short>(V, V.getULL());
      case clang::BuiltinType::UInt:
        return executePrintValue<unsigned int>(V, V.getULL());
      case clang::BuiltinType::ULong:
        return executePrintValue<unsigned long>(V, V.getULL());
      case clang::BuiltinType::ULongLong:
        return executePrintValue<unsigned long long>(V, V.getULL());

      case clang::BuiltinType::Float:
        return executePrintValue<float>(V, V.getFloat());
      case clang::BuiltinType::Double:
        return executePrintValue<double>(V, V.getDouble());
      case clang::BuiltinType::LongDouble:
        return executePrintValue<long double>(V, V.getLongDouble());

      default:
        return executePrintValue<void *>(V, V.getPtr());
    }
  }
  else if (Ty->isIntegralOrEnumerationType()) {
    return executePrintValue<long long>(V, V.getLL());
  }
  else if (Ty->isFunctionType()) {
    return executePrintValue<const void *>(V, &V);
  }
  else if (Ty->isPointerType()
           || Ty->isReferenceType()
           || Ty->isArrayType()) {
    return executePrintValue<void *>(V, V.getPtr());
  }
  else {
    // struct case.
    return executePrintValue<void *>(V, V.getPtr());
  }
}

static std::string printEnumValue(const Value &V) {
  std::stringstream enumString;
  clang::ASTContext &C = V.getASTContext();
  clang::QualType Ty = V.getType().getDesugaredType(C);
  const clang::EnumType *EnumTy = Ty.getNonReferenceType()->getAs<clang::EnumType>();
  assert(EnumTy && "ValuePrinter.cpp: ERROR, printEnumValue invoked for a non enum type.");
  clang::EnumDecl *ED = EnumTy->getDecl();
  uint64_t value = V.getULL();
  bool IsFirst = true;
  llvm::APSInt ValAsAPSInt = C.MakeIntValue(value, Ty);
  for (clang::EnumDecl::enumerator_iterator I = ED->enumerator_begin(),
           E = ED->enumerator_end(); I != E; ++I) {
    if (I->getInitVal() == ValAsAPSInt) {
      if (!IsFirst) {
        enumString << " ? ";
      }
      enumString << "(" << I->getQualifiedNameAsString() << ")";
      IsFirst = false;
    }
  }
  enumString << " : " << printQualType(C, ED->getIntegerType()) << " "
    << ValAsAPSInt.toString(/*Radix = */10);
  return enumString.str();
}

static std::string printFunctionValue(const Value &V, const void *ptr, clang::QualType Ty) {
  std::string functionString;
  llvm::raw_string_ostream o(functionString);
  o << "Function @" << ptr << '\n';

  clang::ASTContext &C = V.getASTContext();
  Interpreter &Interp = *const_cast<Interpreter *>(V.getInterpreter());
  const Transaction *T = Interp.getLastTransaction();
  assert(T->getWrapperFD() && "Must have a wrapper.");
  clang::FunctionDecl *WrapperFD = T->getWrapperFD();

  const clang::FunctionDecl *FD = 0;
  // CE should be the setValueNoAlloc call expr.
  if (const clang::CallExpr *CallE
      = llvm::dyn_cast_or_null<clang::CallExpr>(
          utils::Analyze::GetOrCreateLastExpr(WrapperFD,
              /*foundAtPos*/0,
              /*omitDS*/false,
                                              &Interp.getSema()))) {
    if (const clang::FunctionDecl *FDsetValue
        = llvm::dyn_cast_or_null<clang::FunctionDecl>(CallE->getCalleeDecl())) {
      if (FDsetValue->getNameAsString() == "setValueNoAlloc" &&
          CallE->getNumArgs() == 5) {
        const clang::Expr *Arg4 = CallE->getArg(4);
        while (const clang::CastExpr *CastE
            = clang::dyn_cast<clang::CastExpr>(Arg4))
          Arg4 = CastE->getSubExpr();
        if (const clang::DeclRefExpr *DeclRefExp
            = llvm::dyn_cast<clang::DeclRefExpr>(Arg4))
          FD = llvm::dyn_cast<clang::FunctionDecl>(DeclRefExp->getDecl());
      }
    }
  }

  if (FD) {
    clang::SourceRange SRange = FD->getSourceRange();
    const char *cBegin = 0;
    const char *cEnd = 0;
    bool Invalid;
    if (SRange.isValid()) {
      clang::SourceManager &SM = C.getSourceManager();
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
      const clang::FunctionDecl *FDef;
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
  functionString = o.str();
  return functionString;
}

static std::string printUnpackedClingValue(const Value &V) {
  std::stringstream strm;

  clang::ASTContext &C = V.getASTContext();
  clang::QualType QT = V.getType();
  clang::QualType Ty = QT.getDesugaredType(C).getNonReferenceType();

  if (Ty->isNullPtrType()) {
    // special case nullptr_t
    strm << "nullptr_t";
  } else if (Ty->isEnumeralType()) {
    // special case enum printing, using compiled information
    strm << printEnumValue(V);
  } else if (Ty->isFunctionType()) {
    // special case function printing, using compiled information
    strm << printFunctionValue(V, &V, Ty);
  } else if ((Ty->isPointerType() || Ty->isMemberPointerType()) && Ty->getPointeeType()->isFunctionProtoType()) {
    // special case function printing, using compiled information
    strm << printFunctionValue(V, V.getPtr(), Ty->getPointeeType());
  } else if (clang::CXXRecordDecl *CXXRD = Ty->getAsCXXRecordDecl()) {
    if (CXXRD->isLambda()) {
      strm << "@" << V.getPtr();
    } else {
      // default case, modular printing using cling::printValue
      strm << invokePrintValueOverload(V);
    }
  } else {
    // default case, modular printing using cling::printValue
    strm << invokePrintValueOverload(V);
  }

  return strm.str();
}

namespace cling {

  // General fallback - prints the address
  std::string printValue(const void *ptr) {
    if (!ptr) {
      return "nullptr";
    } else {
      std::ostringstream strm;
      strm << "@" << ptr;
      if (!utils::isAddressValid(ptr))
        strm << " <invalid memory address>";
      return strm.str();
    }
  }

  // void pointer
  std::string printValue(const void **ptr) {
    if (!*ptr) {
      return "nullptr";
    } else {
      std::ostringstream strm;
      strm << *ptr;
      if (!utils::isAddressValid(*ptr))
        strm << " <invalid memory address>";
      return strm.str();
    }
  }

  // Bool
  std::string printValue(const bool *val) {
    return *val ? "true" : "false";
  }

  // Chars
  static std::string printChar(signed char val, bool apostrophe) {
    std::ostringstream strm;
    if (val > 0x1F && val < 0x7F) {
      if (apostrophe)
        strm << "'";
      strm << val;
      if (apostrophe)
        strm << "'";
    } else {
      strm << "0x" << std::hex << (int) val;
    }
    return strm.str();
  }

  std::string printValue(const char *val) {
    return printChar(*val, true);
  }

  std::string printValue(const signed char *val) {
    return printChar(*val, true);
  }

  std::string printValue(const unsigned char *val) {
    return printChar(*val, true);
  }

  // Ints
  std::string printValue(const short *val) {
    std::ostringstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const unsigned short *val) {
    std::ostringstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const int *val) {
    std::ostringstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const unsigned int *val) {
    std::ostringstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const long *val) {
    std::ostringstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const unsigned long *val) {
    std::ostringstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const long long *val) {
    std::ostringstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const unsigned long long *val) {
    std::ostringstream strm;
    strm << *val;
    return strm.str();
  }

  // Reals
  std::string printValue(const float *val) {
    std::ostringstream strm;
    strm << std::showpoint << *val << "f";
    return strm.str();
  }

  std::string printValue(const double *val) {
    std::ostringstream strm;
    strm << std::showpoint << *val;
    return strm.str();
  }

  std::string printValue(const long double *val) {
    std::ostringstream strm;
    strm << *val << "L";
    return strm.str();
  }

  // Char pointers
  std::string printValue(const char *const *val) {
    if (!*val) {
      return "nullptr";
    } else {
      std::ostringstream strm;
      strm << "\"";
      // 10000 limit to prevent potential printing of the whole RAM / inf loop
      for (const char *cobj = *val; *cobj != 0 && cobj - *val < 10000; ++cobj) {
        strm << printChar(*cobj, false);
      }
      strm << "\"";
      return strm.str();
    }
  }

  std::string printValue(const char **val) {
    return printValue((const char *const *) val);
  }

  // std::string
  std::string printValue(const std::string *val) {
    return "\"" + *val + "\"";
  }

  // cling::Value
  std::string printValue(const Value *value) {
    std::ostringstream strm;

    if (!value->isValid()) {
      strm << "<<<invalid>>> @" << value;
    } else {
      clang::ASTContext &C = value->getASTContext();
      clang::QualType QT = value->getType();
      strm << "boxes [";
      strm << "("
      << cling::utils::TypeName::GetFullyQualifiedName(QT, C)
      << ") ";
      if (!QT->isVoidType()) {
        strm << printUnpackedClingValue(*value);
      }
      strm << "]";
    }

    return strm.str();
  }

  namespace valuePrinterInternal {

    std::string printTypeInternal(const Value &V) {
      return printQualType(V.getASTContext(), V.getType());
    }

    std::string printValueInternal(const Value &V) {
      static bool includedRuntimePrintValue = false; // initialized only once as a static function variable
      // Include "RuntimePrintValue.h" only on the first printing.
      // This keeps the interpreter lightweight and reduces the startup time.
      if (!includedRuntimePrintValue) {
        V.getInterpreter()->declare("#include \"cling/Interpreter/RuntimePrintValue.h\"");
        includedRuntimePrintValue = true;
      }
      return printUnpackedClingValue(V);
    }
  } // end namespace valuePrinterInternal
} // end namespace cling
