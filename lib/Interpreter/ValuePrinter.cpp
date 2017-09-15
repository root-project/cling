//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Value.h"

#include "EnterUserCodeRAII.h"

#include "cling/Interpreter/CValuePrinter.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Interpreter/Value.h"
#include "cling/Utils/AST.h"
#include "cling/Utils/Output.h"
#include "cling/Utils/Validation.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/Support/Format.h"
#include "llvm/ExecutionEngine/GenericValue.h"

#include <locale>
#include <string>

// GCC 4.x doesn't have the proper UTF-8 conversion routines. So use the
// LLVM conversion routines (which require a buffer 4x string length).
#if !defined(__GLIBCXX__) || (__GNUC__ >= 5)
 #include <codecvt>
#else
 #define LLVM_UTF8
 #include "llvm/Support/ConvertUTF.h"
#endif

using namespace cling;

// Implements the CValuePrinter interface.
extern "C" void cling_PrintValue(void * /*cling::Value**/ V) {
  //Value* value = (Value*)V;

  //std::string typeStr = printTypeInternal(*value);
  //std::string valueStr = printValueInternal(*value);
}

// Exported for RuntimePrintValue.h
namespace cling {
  namespace valuePrinterInternal {
    extern const char* const kEmptyCollection = "{}";
  }
}

namespace {

const static char
  * const kNullPtrStr = "nullptr",
  * const kNullPtrTStr = "nullptr_t",
  * const kTrueStr = "true",
  * const kFalseStr = "false",
  * const kInvalidAddr = " <invalid memory address>";

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

static clang::QualType
getElementTypeAndExtent(const clang::ConstantArrayType *CArrTy,
                        std::string& extent) {
  clang::QualType ElementTy = CArrTy->getElementType();
  const llvm::APInt &APSize = CArrTy->getSize();
  extent += '[' +  std::to_string(APSize.getZExtValue()) + ']';
  if (auto CArrElTy
      = llvm::dyn_cast<clang::ConstantArrayType>(ElementTy.getTypePtr())) {
    return getElementTypeAndExtent(CArrElTy, extent);
  }
  return ElementTy;
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
    if (Ty->isConstantArrayType()) {
      std::string extent("(*)");
      clang::QualType InnermostElTy
        = getElementTypeAndExtent(C.getAsConstantArrayType(Ty), extent);
      return enclose(InnermostElTy, C, "(", (extent + ")*(void**)").c_str());
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
    LangOpts.AccessControl = false;
  }

  ~AccessCtrlRAII_t() {
    LangOpts.AccessControl = savedAccessControl;
  }

};

#ifndef NDEBUG
/// Is typenam parsable?
bool canParseTypeName(cling::Interpreter& Interp, llvm::StringRef typenam) {

  AccessCtrlRAII_t AccessCtrlRAII(Interp);
  LockCompilationDuringUserCodeExecutionRAII LCDUCER(Interp);

  cling::Interpreter::CompilationResult Res
    = Interp.declare("namespace { void* cling_printValue_Failure_Typename_check"
                     " = (void*)" + typenam.str() + "nullptr; }");
  if (Res != cling::Interpreter::kSuccess)
    cling::errs() << "ERROR in cling::canParseTypeName(): "
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

static std::string printAddress(const void* Ptr, const char Prfx = 0) {
  if (!Ptr)
    return kNullPtrStr;

  cling::smallstream Strm;
  if (Prfx)
    Strm << Prfx;
  Strm << Ptr;
  if (!utils::isAddressValid(Ptr))
    Strm << kInvalidAddr;
  return Strm.str();
}

} // anonymous namespace

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
  static std::string printOneChar(char Val,
                                  const std::locale& Locale = std::locale()) {
    llvm::SmallString<128> Buf;
    llvm::raw_svector_ostream Strm(Buf);
    Strm << "'";
    if (!std::isprint(Val, Locale)) {
      switch (std::isspace(Val, Locale) ? Val : 0) {
        case '\t': Strm << "\\t"; break;
        case '\n': Strm << "\\n"; break;
        case '\r': Strm << "\\r"; break;
        case '\f': Strm << "\\f"; break;
        case '\v': Strm << "\\v"; break;
        default:
          Strm << llvm::format_hex(uint64_t(Val)&0xff, 4);
      }
    }
    else
      Strm << Val;
    Strm << "'";
    return Strm.str();
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
    cling::smallstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const unsigned short *val) {
    cling::smallstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const int *val) {
    cling::smallstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const unsigned int *val) {
    cling::smallstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const long *val) {
    cling::smallstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const unsigned long *val) {
    cling::smallstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const long long *val) {
    cling::smallstream strm;
    strm << *val;
    return strm.str();
  }

  std::string printValue(const unsigned long long *val) {
    cling::smallstream strm;
    strm << *val;
    return strm.str();
  }

  // Reals
  std::string printValue(const float *val) {
    cling::smallstream strm;
    strm << llvm::format("%.5f", *val) << 'f';
    return strm.str();
  }

  std::string printValue(const double *val) {
    cling::smallstream strm;
    strm << llvm::format("%.6f", *val);
    return strm.str();
  }

  std::string printValue(const long double *val) {
    cling::smallstream strm;
    strm << llvm::format("%.8Lf", *val) << 'L';
    //strm << llvm::format("%Le", *val) << 'L';
    return strm.str();
  }

  // Char pointers
  std::string printString(const char *const *Ptr, size_t N = 10000) {
    // Assumption is this is a string.
    // N is limit to prevent endless loop if Ptr is not really a string.

    const char* Start = *Ptr;
    if (!Start)
      return kNullPtrStr;

    const char* End = Start + N;
    bool IsValid = utils::isAddressValid(Start);
    if (IsValid) {
      // If we're gonnd do this, better make sure the end is valid too
      // FIXME: getpagesize() & GetSystemInfo().dwPageSize might be better
      enum { PAGE_SIZE = 1024 };
      while (!(IsValid = utils::isAddressValid(End)) && N > 1024) {
        N -= PAGE_SIZE;
        End = Start + N;
      }
    }
    if (!IsValid) {
      cling::smallstream Strm;
      Strm << static_cast<const void*>(Start) << kInvalidAddr;
      return Strm.str();
    }

    if (*Start == 0)
      return "\"\"";

    // Copy the bytes until we get a null-terminator
    llvm::SmallString<1024> Buf;
    llvm::raw_svector_ostream Strm(Buf);
    Strm << "\"";
    while (Start < End && *Start)
      Strm << *Start++;
    Strm << "\"";

    return Strm.str();
  }

  std::string printValue(const char *const *val) {
    return printString(val);
  }

  std::string printValue(const char **val) {
    return printString(val);
  }

  // std::string
  std::string printValue(const std::string *val) {
    return "\"" + *val + "\"";
  }

  static std::string quoteString(std::string Str, const char Prefix) {
    // No wrap
    if (!Prefix)
      return Str;
    // Quoted wrap
    if (Prefix==1)
      return enclose(std::move(Str), "\"", "\"", 2);

    // Prefix quoted wrap
    char Begin[3] = { Prefix, '"', 0 };
    return enclose(std::move(Str), Begin, &Begin[1], 3);
  }

  static std::string quoteString(const char* Str, size_t N, const char Prefix) {
    return quoteString(std::string(Str, Str[N-1] == 0 ? (N-1) : N), Prefix);
  }

#ifdef LLVM_UTF8
  using llvm::ConversionResult;
  using llvm::ConversionFlags;
  using llvm::lenientConversion;
  using llvm::UTF8;
  using llvm::UTF16;
  using llvm::UTF32;
  template <class T> struct CharTraits;
  template <> struct CharTraits<char16_t> {
    static ConversionResult convert(const char16_t** begin, const char16_t* end,
                                    char** d, char* dEnd, ConversionFlags F ) {
      return ConvertUTF16toUTF8(reinterpret_cast<const UTF16**>(begin),
                                reinterpret_cast<const UTF16*>(end),
                                reinterpret_cast<UTF8**>(d),
                                reinterpret_cast<UTF8*>(dEnd), F);
    }
  };
  template <> struct CharTraits<char32_t> {
    static ConversionResult convert(const char32_t** begin, const char32_t* end,
                                    char** d, char* dEnd, ConversionFlags F ) {
      return ConvertUTF32toUTF8(reinterpret_cast<const UTF32**>(begin),
                                reinterpret_cast<const UTF32*>(end),
                                reinterpret_cast<UTF8**>(d),
                                reinterpret_cast<UTF8*>(dEnd), F);
    }
  };
  template <> struct CharTraits<wchar_t> {
    static ConversionResult convert(const wchar_t** src, const wchar_t* srcEnd,
                                    char** dst, char* dEnd, ConversionFlags F) {
      switch (sizeof(wchar_t)) {
        case sizeof(char16_t):
          return CharTraits<char16_t>::convert(
                            reinterpret_cast<const char16_t**>(src),
                            reinterpret_cast<const char16_t*>(srcEnd),
                            dst, dEnd, F);
        case sizeof(char32_t):
          return CharTraits<char32_t>::convert(
                            reinterpret_cast<const char32_t**>(src),
                            reinterpret_cast<const char32_t*>(srcEnd),
                            dst, dEnd, F);
        default: break;
      }
      llvm_unreachable("wchar_t conversion failure");
    }
  };

  template <typename T>
  static std::string encodeUTF8(const T* const Str, size_t N, const char Prfx) {
    const T *Bgn = Str,
            *End = Str + N;
    std::string Result;
    Result.resize(UNI_MAX_UTF8_BYTES_PER_CODE_POINT * N);
    char *ResultPtr = &Result[0],
         *ResultEnd = ResultPtr + Result.size();

    CharTraits<T>::convert(&Bgn, End, &ResultPtr, ResultEnd, lenientConversion);
    Result.resize(ResultPtr - &Result[0]);
    return quoteString(std::move(Result), Prfx);
  }

#else // !LLVM_UTF8

  template <class T> struct CharTraits { typedef T value_type; };
#if defined(LLVM_ON_WIN32) // Likely only to be needed when _MSC_VER < 19??
  template <> struct CharTraits<char16_t> { typedef unsigned short value_type; };
  template <> struct CharTraits<char32_t> { typedef unsigned int value_type; };
#endif

  template <typename T>
  static std::string encodeUTF8(const T* const Str, size_t N, const char Prfx) {
    typedef typename CharTraits<T>::value_type value_type;
    std::wstring_convert<std::codecvt_utf8_utf16<value_type>, value_type> Convert;
    const value_type* Src = reinterpret_cast<const value_type*>(Str);
    return quoteString(Convert.to_bytes(Src, Src + N), Prfx);
  }
#endif // LLVM_UTF8

  template <typename T>
  std::string utf8Value(const T* const Str, size_t N, const char Prefix,
                        std::string (*Func)(const T* const Str, size_t N,
                        const char Prfx) ) {
    if (!Str)
      return kNullPtrStr;
    if (N==0)
      return printAddress(Str, '@');

    // Drop the null terminator or else it will be encoded into the std::string.
    return Func(Str, Str[N-1] == 0 ? (N-1) : N, Prefix);
  }

  // declaration: cling/Utils/UTF8.h & cling/Interpreter/RuntimePrintValue.h
  template <class T>
  std::string toUTF8(const T* const Str, size_t N, const char Prefix);

  template <>
  std::string toUTF8<char16_t>(const char16_t* const Str, size_t N,
                               const char Prefix) {
    return utf8Value(Str, N, Prefix, encodeUTF8);
  }

  template <>
  std::string toUTF8<char32_t>(const char32_t* const Str, size_t N,
                               const char Prefix) {
    return utf8Value(Str, N, Prefix, encodeUTF8);
  }

  template <>
  std::string toUTF8<wchar_t>(const wchar_t* const Str, size_t N,
                              const char Prefix) {
    static_assert(sizeof(wchar_t) == sizeof(char16_t) ||
                  sizeof(wchar_t) == sizeof(char32_t), "Bad wchar_t size");

    if (sizeof(wchar_t) == sizeof(char32_t))
      return toUTF8(reinterpret_cast<const char32_t * const>(Str), N, Prefix);

    return toUTF8(reinterpret_cast<const char16_t * const>(Str), N, Prefix);
  }

  template <>
  std::string toUTF8<char>(const char* const Str, size_t N, const char Prefix) {
    return utf8Value(Str, N, Prefix, quoteString);
  }

  template <typename T>
  static std::string toUTF8(
      const std::basic_string<T, std::char_traits<T>, std::allocator<T>>* Src,
      const char Prefix) {
    if (!Src)
      return kNullPtrStr;
    return encodeUTF8(Src->data(), Src->size(), Prefix);
  }

  std::string printValue(const std::u16string* Val) {
    return toUTF8(Val, 'u');
  }

  std::string printValue(const std::u32string* Val) {
    return toUTF8(Val, 'U');
  }

  std::string printValue(const std::wstring* Val) {
    return toUTF8(Val, 'L');
  }

  // Unicode chars
  template <typename T>
  static std::string toUnicode(const T* Src, const char Prefix, char Esc = 0) {
    if (!Src)
      return kNullPtrStr;
    if (!Esc)
      Esc = Prefix;

    llvm::SmallString<128> Buf;
    llvm::raw_svector_ostream Strm(Buf);
    Strm << Prefix << "'\\" << Esc
         << llvm::format_hex_no_prefix(unsigned(*Src), sizeof(T)*2) << "'";
    return Strm.str();
  }

  std::string printValue(const char16_t *Val) {
    return toUnicode(Val, 'u');
  }

  std::string printValue(const char32_t *Val) {
    return toUnicode(Val, 'U');
  }

  std::string printValue(const wchar_t *Val) {
    return toUnicode(Val, 'L', 'x');
  }
} // end namespace cling

namespace {

static std::string callPrintValue(const Value& V, const void* Val) {
  Interpreter *Interp = V.getInterpreter();
  Value printValueV;

  {
    // Use an llvm::raw_ostream to prepend '0x' in front of the pointer value.

    cling::ostrstream Strm;
    Strm << "cling::printValue(";
    Strm << getTypeString(V);
    Strm << &Val;
    Strm << ");";

    // We really don't care about protected types here (ROOT-7426)
    AccessCtrlRAII_t AccessCtrlRAII(*Interp);
    LockCompilationDuringUserCodeExecutionRAII LCDUCER(*Interp);
    Interp->evaluate(Strm.str(), printValueV);
  }

  if (printValueV.isValid() && printValueV.getPtr())
    return *(std::string *) printValueV.getPtr();

  // That didn't work. We probably diagnosed the issue as part of evaluate().
  cling::errs() <<"ERROR in cling's callPrintValue(): cannot pass value!\n";

  // Check that the issue comes from an unparsable type name: lambdas, unnamed
  // namespaces, types declared inside functions etc. Assert on everything
  // else.
  assert(!canParseTypeName(*Interp, getTypeString(V))
         && "printValue failed on a valid type name.");

  return "ERROR in cling's callPrintValue(): missing value string.";
}

template <typename T>
class HasExplicitPrintValue {
  template <typename C,
            typename = decltype(cling::printValue((C*)nullptr))>
  static std::true_type  test(int);
  static std::false_type test(...);
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T> static
typename std::enable_if<!HasExplicitPrintValue<const T>::value, std::string>::type
executePrintValue(const Value& V, const T& val) {
  return callPrintValue(V, &val);
}

template <typename T> static
typename std::enable_if<HasExplicitPrintValue<const T>::value, std::string>::type
executePrintValue(const Value& V, const T& val) {
  return printValue(&val);
}


static std::string printEnumValue(const Value &V) {
  cling::ostrstream enumString;
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
  cling::largestream o;
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
  return o.str();
}

static std::string printStringType(const Value &V, const clang::Type* Type) {
  switch (V.getInterpreter()->getLookupHelper().getStringType(Type)) {
    case LookupHelper::kStdString:
      return executePrintValue<std::string>(V, *(std::string*)V.getPtr());
    case LookupHelper::kWCharString:
      return executePrintValue<std::wstring>(V, *(std::wstring*)V.getPtr());
    case LookupHelper::kUTF16Str:
      return executePrintValue<std::u16string>(V, *(std::u16string*)V.getPtr());
    case LookupHelper::kUTF32Str:
      return executePrintValue<std::u32string>(V, *(std::u32string*)V.getPtr());
    default:
      break;
  }
  return "";
}

static std::string printUnpackedClingValue(const Value &V) {
  // Find the Type for `std::string`. We are guaranteed to have that declared
  // when this function is called; RuntimePrintValue.h #includes it.
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

    std::string Str = printStringType(V, CXXRD->getTypeForDecl());
    if (!Str.empty())
      return Str;
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
  return callPrintValue(V, V.getPtr());
}

} // anonymous namespace

namespace cling {
  // cling::Value
  std::string printValue(const Value *value) {
    cling::smallstream strm;

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
        Interpreter* Interp = V.getInterpreter();
        LockCompilationDuringUserCodeExecutionRAII LCDUCER(*Interp);
        Interp->declare("#include \"cling/Interpreter/RuntimePrintValue.h\"");
        includedRuntimePrintValue = true;
      }
      return printUnpackedClingValue(V);
    }
  } // end namespace valuePrinterInternal
} // end namespace cling
