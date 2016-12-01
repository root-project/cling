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

// Exported for RuntimePrintValue.h
namespace cling {
  namespace valuePrinterInternal {
    const char* const kEmptyCollection = "{}";
  }
}

namespace {

const static char
  * const kNullPtrStr = "nullptr",
  * const kNullPtrTStr = "nullptr_t",
  * const kTrueStr = "true",
  * const kFalseStr = "false";

static std::string enclose(std::string Mid, const char* Begin,
                           const char* End, size_t Hint = 0) {
  Mid.reserve(Mid.size() + Hint ? Hint : (::strlen(Begin) + ::strlen(End)));
  Mid.insert(0, Begin);
  Mid.append(End);
  return Mid;
}

static std::string enclose(const clang::QualType& Ty, clang::ASTContext& C,
                           const char* Begin = "(", const char* End = "*)",
                           size_t Hint = 3) {
  return enclose(cling::utils::TypeName::GetFullyQualifiedName(Ty, C),
                 Begin, End, Hint);
}

static std::string getTypeString(const Value &V) {
  clang::ASTContext &C = V.getASTContext();
  clang::QualType Ty = V.getType().getDesugaredType(C).getNonReferenceType();

  if (llvm::dyn_cast<clang::BuiltinType>(Ty.getCanonicalType()))
    return enclose(Ty, C);

  if (Ty->isPointerType()) {
    // Print char pointers as strings.
    if (Ty->getPointeeType()->isCharType())
      return enclose(Ty, C);

    // Fallback to void pointer for other pointers and print the address.
    return "(const void**)";
  }
  if (Ty->isArrayType()) {
    const clang::ArrayType *ArrTy = Ty->getAsArrayTypeUnsafe();
    clang::QualType ElementTy = ArrTy->getElementType();
    // In case of char ElementTy, printing as string
    if (ElementTy->isCharType())
      return "(const char **)";

    if (Ty->isConstantArrayType()) {
      const clang::ConstantArrayType *CArrTy = C.getAsConstantArrayType(Ty);
      const llvm::APInt &APSize = CArrTy->getSize();

      // typeWithOptDeref example for int[40] array: "((int(*)[40])*(void**)0x5c8f260)"
      return enclose(ElementTy, C, "(", "(*)[", 5) +
                     std::to_string(APSize.getZExtValue()) + "])*(void**)";
    }
    return "(void**)";
  }
  if (Ty->isObjCObjectPointerType())
    return "(const void**)";

  // In other cases, dereference the address of the object.
  // If no overload or specific template matches,
  // the general template will be used which only prints the address.
  return enclose(Ty, C, "*(", "**)", 5);
}

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

static std::string printDeclType(const clang::QualType& QT,
                                 const clang::NamedDecl* D) {
  if (!QT.hasQualifiers())
    return D->getQualifiedNameAsString();
  return QT.getQualifiers().getAsString() + " " + D->getQualifiedNameAsString();
}

static std::string printQualType(clang::ASTContext& Ctx, clang::QualType QT) {
  using namespace clang;
  const QualType QTNonRef = QT.getNonReferenceType();

  std::string ValueTyStr("(");
  if (const TagType *TTy = dyn_cast<TagType>(QTNonRef))
    ValueTyStr += printDeclType(QTNonRef, TTy->getDecl());
  else if (const RecordType *TRy = dyn_cast<RecordType>(QTNonRef))
    ValueTyStr += printDeclType(QTNonRef, TRy->getDecl());
  else {
    const QualType QTCanon = QTNonRef.getCanonicalType();
    if (QTCanon->isBuiltinType() && !QTNonRef->isFunctionPointerType()
        && !QTNonRef->isMemberPointerType()) {
      ValueTyStr += QTCanon.getAsString(Ctx.getPrintingPolicy());
    }
    else if (const TypedefType* TDTy = dyn_cast<TypedefType>(QTNonRef)) {
      // FIXME: TemplateSpecializationType & SubstTemplateTypeParmType checks are
      // predominately to get STL containers to print nicer and might be better
      // handled in GetFullyQualifiedName.
      //
      // std::vector<Type>::iterator is a TemplateSpecializationType
      // std::vector<Type>::value_type is a SubstTemplateTypeParmType
      //
      QualType SSDesugar = TDTy->getLocallyUnqualifiedSingleStepDesugaredType();
      if (dyn_cast<SubstTemplateTypeParmType>(SSDesugar))
        ValueTyStr += utils::TypeName::GetFullyQualifiedName(QTCanon, Ctx);
      else if (dyn_cast<TemplateSpecializationType>(SSDesugar))
        ValueTyStr += utils::TypeName::GetFullyQualifiedName(QTNonRef, Ctx);
      else
        ValueTyStr += printDeclType(QTNonRef, TDTy->getDecl());
    }
    else
      ValueTyStr += utils::TypeName::GetFullyQualifiedName(QTNonRef, Ctx);
  }

  if (QT->isReferenceType())
    ValueTyStr += " &";

  return ValueTyStr + ")";
}

} // anonymous namespace

template<typename T>
static std::string executePrintValue(const Value &V, const T &val) {
  Interpreter *Interp = V.getInterpreter();
  Value printValueV;

  {
    // Use an llvm::raw_ostream to prepend '0x' in front of the pointer value.

    llvm::SmallString<512> Buf;
    llvm::raw_svector_ostream Strm(Buf);
    Strm << "cling::printValue(";
    Strm << getTypeString(V);
    Strm << (const void*) &val;
    Strm << ");";

    // We really don't care about protected types here (ROOT-7426)
    AccessCtrlRAII_t AccessCtrlRAII(*Interp);
    clang::DiagnosticsEngine& Diag = Interp->getCI()->getDiagnostics();
    bool oldSuppDiags = Diag.getSuppressAllDiagnostics();
    Diag.setSuppressAllDiagnostics(true);
    Interp->evaluate(Strm.str(), printValueV);
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
  o << "Function @" << ptr;

  // If a function is the first thing printed in a session,
  // getLastTransaction() will point to the transaction that loaded the
  // ValuePrinter, and won't have a wrapper FD.
  // Even if it did have one it wouldn't be the one that was requested to print.

  Interpreter &Interp = *const_cast<Interpreter *>(V.getInterpreter());
  const Transaction *T = Interp.getLastTransaction();
  if (clang::FunctionDecl *WrapperFD = T->getWrapperFD()) {
    clang::ASTContext &C = V.getASTContext();
    const clang::FunctionDecl *FD = nullptr;
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
      o << '\n';
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
      // type-based print() never and decl-based print() sometimes does not
      // include a final newline:
      o << '\n';
    }
  }
  functionString = o.str();
  return functionString;
}

static std::string printAddress(const void* Ptr, const char Prfx = 0) {
  if (!Ptr)
    return kNullPtrStr;

  llvm::SmallString<256> Buf;
  llvm::raw_svector_ostream Strm(Buf);
  if (Prfx)
    Strm << Prfx;
  Strm << Ptr;
  if (!utils::isAddressValid(Ptr))
    Strm << " <invalid memory address>";
  return Strm.str();
}

static std::string printUnpackedClingValue(const Value &V) {
  const clang::ASTContext &C = V.getASTContext();
  const clang::QualType Td = V.getType().getDesugaredType(C);
  const clang::QualType Ty = Td.getNonReferenceType();

  if (Ty->isNullPtrType()) {
    // special case nullptr_t
    return kNullPtrTStr;
  } else if (Ty->isEnumeralType()) {
    // special case enum printing, using compiled information
    return printEnumValue(V);
  } else if (Ty->isFunctionType()) {
    // special case function printing, using compiled information
    return printFunctionValue(V, &V, Ty);
  } else if ((Ty->isPointerType() || Ty->isMemberPointerType()) && Ty->getPointeeType()->isFunctionProtoType()) {
    // special case function printing, using compiled information
    return printFunctionValue(V, V.getPtr(), Ty->getPointeeType());
  } else if (clang::CXXRecordDecl *CXXRD = Ty->getAsCXXRecordDecl()) {
    if (CXXRD->isLambda())
      return printAddress(V.getPtr(), '@');
  } else if (const clang::BuiltinType *BT
      = llvm::dyn_cast<clang::BuiltinType>(Td.getCanonicalType().getTypePtr())) {
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
        break;
    }
  } else
    assert(!Ty->isIntegralOrEnumerationType() && "Bad Type.");

  if (!V.getPtr())
    return kNullPtrStr;

  // Print all the other cases by calling into runtime 'cling::printValue()'.
  // Ty->isPointerType() || Ty->isReferenceType() || Ty->isArrayType()
  // Ty->isObjCObjectPointerType()
  return executePrintValue<void*>(V, V.getPtr());
}

namespace cling {

  // General fallback - prints the address
  std::string printValue(const void *ptr) {
    return printAddress(ptr, '@');
  }

  // void pointer
  std::string printValue(const void **ptr) {
    return printAddress(*ptr);
  }

  // Bool
  std::string printValue(const bool *val) {
    return *val ? kTrueStr : kFalseStr;
  }

  // Chars
  static void printChar(signed char val, std::ostringstream& strm) {
    if (val > 0x1F && val < 0x7F) {
      strm << val;
    } else {
      std::ios::fmtflags prevFlags = strm.flags();
      strm << "0x" << std::hex << (int) val;
      strm.flags(prevFlags);
    }
  }

  static std::string printOneChar(signed char val) {
    std::ostringstream strm;
    strm << "'";
    printChar(val, strm);
    strm << "'";
    return strm.str();
  }

  std::string printValue(const char *val) {
    return printOneChar(*val);
  }

  std::string printValue(const signed char *val) {
    return printOneChar(*val);
  }

  std::string printValue(const unsigned char *val) {
    return printOneChar(*val);
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
      return kNullPtrStr;
    } else {
      std::ostringstream strm;
      strm << "\"";
      // 10000 limit to prevent potential printing of the whole RAM / inf loop
      for (const char *cobj = *val; *cobj != 0 && cobj - *val < 10000; ++cobj) {
        printChar(*cobj, strm);
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

    if (value->isValid()) {
      clang::ASTContext &C = value->getASTContext();
      clang::QualType QT = value->getType();
      strm << "boxes [";
      strm << enclose(QT, C, "(", ") ", 3);
      if (!QT->isVoidType()) {
        strm << printUnpackedClingValue(*value);
      }
      strm << "]";
    } else
      strm << "<<<invalid>>> " << printAddress(value, '@');

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
