//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Value.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"
#include "cling/Utils/Output.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/CanonicalType.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"

#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_os_ostream.h"

#include <iostream>
#include <sstream>
#include <locale>

#ifdef LLVM_ON_WIN32
#include <Shlwapi.h>
#define strcasestr StrStrIA
#pragma comment(lib, "Shlwapi.lib")
#endif

namespace {

  ///\brief The allocation starts with this layout; it is followed by the
  ///  value's object at m_Payload. This class does not inherit from
  ///  llvm::RefCountedBase because deallocation cannot use this type but must
  ///  free the character array.

  class AllocatedValue {
  public:
    typedef void (*DtorFunc_t)(void*);

  private:
    ///\brief The reference count - once 0, this object will be deallocated.
    mutable unsigned m_RefCnt;

    ///\brief The destructor function.
    DtorFunc_t m_DtorFunc;

    ///\brief The size of the allocation (for arrays)
    unsigned long m_AllocSize;

    ///\brief The number of elements in the array
    unsigned long m_NElements;

    ///\brief The start of the allocation.
    char m_Payload[1];

    static DtorFunc_t PtrToFunc(void* ptr) {
      union {
        void* m_Ptr;
        DtorFunc_t m_Func;
      };
      m_Ptr = ptr;
      return m_Func;
    }


  public:
    ///\brief Initialize the storage management part of the allocated object.
    ///  The allocator is referencing it, thus initialize m_RefCnt with 1.
    ///\param [in] dtorFunc - the function to be called before deallocation.
    AllocatedValue(void* dtorFunc, size_t allocSize, size_t nElements):
      m_RefCnt(1), m_DtorFunc(PtrToFunc(dtorFunc)), m_AllocSize(allocSize),
      m_NElements(nElements)
    {}

    char* getPayload() { return m_Payload; }

    static unsigned getPayloadOffset() {
      static const AllocatedValue Dummy(0,0,0);
      return Dummy.m_Payload - (const char*)&Dummy;
    }

    static AllocatedValue* getFromPayload(void* payload) {
      return
        reinterpret_cast<AllocatedValue*>((char*)payload - getPayloadOffset());
    }

    void Retain() { ++m_RefCnt; }

    ///\brief This object must be allocated as a char array. Deallocate it as
    ///   such.
    void Release() {
      assert (m_RefCnt > 0 && "Reference count is already zero.");
      if (--m_RefCnt == 0) {
        if (m_DtorFunc) {
          char* payload = getPayload();
          for (size_t el = 0; el < m_NElements; ++el)
            (*m_DtorFunc)(payload + el * m_AllocSize / m_NElements);
        }
        delete [] (char*)this;
      }
    }
  };
}

namespace cling {

  Value::Value(const Value& other):
    m_Storage(other.m_Storage), m_StorageType(other.m_StorageType),
    m_Type(other.m_Type), m_Interpreter(other.m_Interpreter) {
    if (other.needsManagedAllocation())
      AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Retain();
  }

  Value::Value(clang::QualType clangTy, Interpreter& Interp):
    m_StorageType(determineStorageType(clangTy)),
    m_Type(clangTy.getAsOpaquePtr()),
    m_Interpreter(&Interp) {
    if (needsManagedAllocation())
      ManagedAllocate();
  }

  Value& Value::operator =(const Value& other) {
    // Release old value.
    if (needsManagedAllocation())
      AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Release();

    // Retain new one.
    m_Type = other.m_Type;
    m_Storage = other.m_Storage;
    m_StorageType = other.m_StorageType;
    m_Interpreter = other.m_Interpreter;
    if (needsManagedAllocation())
      AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Retain();
    return *this;
  }

  Value& Value::operator =(Value&& other) {
    // Release old value.
    if (needsManagedAllocation())
      AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Release();

    // Move new one.
    m_Type = other.m_Type;
    m_Storage = other.m_Storage;
    m_StorageType = other.m_StorageType;
    m_Interpreter = other.m_Interpreter;
    // Invalidate other so it will not release.
    other.m_StorageType = kUnsupportedType;

    return *this;
  }

  Value::~Value() {
    if (needsManagedAllocation())
      AllocatedValue::getFromPayload(m_Storage.m_Ptr)->Release();
  }

  clang::QualType Value::getType() const {
    return clang::QualType::getFromOpaquePtr(m_Type);
  }

  clang::ASTContext& Value::getASTContext() const {
    return m_Interpreter->getCI()->getASTContext();
  }

  bool Value::isValid() const { return !getType().isNull(); }

  bool Value::isVoid() const {
    const clang::ASTContext& Ctx = getASTContext();
    return isValid() && Ctx.hasSameType(getType(), Ctx.VoidTy);
  }

  size_t Value::GetNumberOfElements() const {
    if (const clang::ConstantArrayType* ArrTy
        = llvm::dyn_cast<clang::ConstantArrayType>(getType())) {
      llvm::APInt arrSize(sizeof(size_t)*8, 1);
      do {
        arrSize *= ArrTy->getSize();
        ArrTy = llvm::dyn_cast<clang::ConstantArrayType>(ArrTy->getElementType()
                                                         .getTypePtr());
      } while (ArrTy);
      return static_cast<size_t>(arrSize.getZExtValue());
    }
    return 1;
  }

  Value::EStorageType Value::determineStorageType(clang::QualType QT) {
    const clang::Type* desugCanon = QT.getCanonicalType().getTypePtr();
    if (desugCanon->isSignedIntegerOrEnumerationType())
      return kSignedIntegerOrEnumerationType;
    else if (desugCanon->isUnsignedIntegerOrEnumerationType())
      return kUnsignedIntegerOrEnumerationType;
    else if (desugCanon->isRealFloatingType()) {
      const clang::BuiltinType* BT = desugCanon->getAs<clang::BuiltinType>();
      if (BT->getKind() == clang::BuiltinType::Double)
        return kDoubleType;
      else if (BT->getKind() == clang::BuiltinType::Float)
        return kFloatType;
      else if (BT->getKind() == clang::BuiltinType::LongDouble)
        return kLongDoubleType;
    } else if (desugCanon->isPointerType() || desugCanon->isObjectType()
               || desugCanon->isReferenceType()) {
      if (desugCanon->isRecordType() || desugCanon->isConstantArrayType()
          || desugCanon->isMemberPointerType())
        return kManagedAllocation;
      return kPointerType;
    }
    return kUnsupportedType;
  }

  void Value::ManagedAllocate() {
    assert(needsManagedAllocation() && "Does not need managed allocation");
    void* dtorFunc = 0;
    clang::QualType DtorType = getType();
    // For arrays we destruct the elements.
    if (const clang::ConstantArrayType* ArrTy
        = llvm::dyn_cast<clang::ConstantArrayType>(DtorType.getTypePtr())) {
      DtorType = ArrTy->getElementType();
    }
    if (const clang::RecordType* RTy = DtorType->getAs<clang::RecordType>())
      dtorFunc = m_Interpreter->compileDtorCallFor(RTy->getDecl());

    const clang::ASTContext& ctx = getASTContext();
    unsigned payloadSize = ctx.getTypeSizeInChars(getType()).getQuantity();
    char* alloc = new char[AllocatedValue::getPayloadOffset() + payloadSize];
    AllocatedValue* allocVal = new (alloc) AllocatedValue(dtorFunc, payloadSize,
                                                          GetNumberOfElements());
    m_Storage.m_Ptr = allocVal->getPayload();
  }

  void Value::AssertOnUnsupportedTypeCast() const {
    assert("unsupported type in Value, cannot cast simplistically!" && 0);
  }

  namespace valuePrinterInternal {
    std::string printTypeInternal(const Value& V);
    std::string printValueInternal(const Value& V);
  } // end namespace valuePrinterInternal

  void Value::print(llvm::raw_ostream& Out) const {

    // Get the default type string representation
    std::string typeStr = cling::valuePrinterInternal::printTypeInternal(*this);
    // Get the value string representation, by printValue() method overloading
    std::string valueStr = cling::valuePrinterInternal::printValueInternal(*this);

    // Print the type and the value:
    Out << typeStr << " " << valueStr << '\n';
  }

namespace utf8 {
  // Adapted from utf8++
  enum {
    LEAD_SURROGATE_MIN  = 0xd800u,
    TRAIL_SURROGATE_MIN = 0xdc00u,
    TRAIL_SURROGATE_MAX = 0xdfffu,
    LEAD_OFFSET         = LEAD_SURROGATE_MIN - (0x10000 >> 10),
    CODE_POINT_MAX      = 0x0010ffffu
  };

  static uint8_t mask8(char Octet) {
    return static_cast<uint8_t>(Octet); //  & 0xff
  }

  static bool validate(const char* Str, size_t N,
                       const std::locale& Loc, bool& isPrint) {
    for (size_t i = 0, N1 = (N-1); i < N; ++i) {
      uint8_t n;
      uint8_t C = mask8(Str[i]);
      isPrint = isPrint ? std::isprint(Str[i], Loc) : false;

      if (C <= 0x7f)
        n = 0; // 0bbbbbbb
      else if ((C & 0xe0) == 0xc0)
        n = 1; // 110bbbbb
      else if ( C==0xed && i < N1 && (mask8(Str[i+1]) & 0xa0) == 0xa0)
        return false; //U+d800 to U+dfff
      else if ((C & 0xf0) == 0xe0)
        n = 2; // 1110bbbb
      else if ((C & 0xf8) == 0xf0)
        n = 3; // 11110bbb
#if 0 // unnecessary in 4 byte UTF-8
      else if ((C & 0xfc) == 0xf8)
        n = 4; // 111110bb //byte 5
      else if ((C & 0xfe) == 0xfc) n = 5;
        // 1111110b //byte 6
#endif
      else
        return false;

      // n bytes matching 10bbbbbb follow ?
      for (uint8_t j = 0; j < n && i < N; ++j) {
        if (++i == N)
          return false;
        C = mask8(Str[i]);
        if ((C & 0xc0) != 0x80)
          return false;
      }
    }
    return true;
  }

  static uint8_t sequenceLength(const char* Ptr) {
    uint8_t lead = mask8(*Ptr);
    if (lead < 0x80)
      return 1;
    else if ((lead >> 5) == 0x6)
      return 2;
    else if ((lead >> 4) == 0xe)
      return 3;
    else if ((lead >> 3) == 0x1e)
      return 4;
    return 0;
  }

  static char32_t next(const char*& Ptr) {
    char32_t CP = mask8(*Ptr);
    switch (sequenceLength(Ptr)) {
      case 1: break;
      case 2:
        ++Ptr;
        CP = ((CP << 6) & 0x7ff) + ((*Ptr) & 0x3f);
        break;
      case 3:
        ++Ptr;
        CP = ((CP << 12) & 0xffff) + ((mask8(*Ptr) << 6) & 0xfff);
        ++Ptr;
        CP += (*Ptr) & 0x3f;
        break;
      case 4:
        ++Ptr;
        CP = ((CP << 18) & 0x1fffff) + ((mask8(*Ptr) << 12) & 0x3ffff);
        ++Ptr;
        CP += (mask8(*Ptr) << 6) & 0xfff;
        ++Ptr;
        CP += (*Ptr) & 0x3f;
        break;
    }
    ++Ptr;
    return CP;
  }

  // mimic isprint() for Unicode codepoints
  static bool isPrint(char32_t CP, const std::locale&) {
    // C0
    if (CP <= 0x1F || CP == 0x7F)
      return false;

    // C1
    if (CP >= 0x80 && CP <= 0x9F)
      return false;

    // line/paragraph separators
    if (CP == 0x2028 || CP == 0x2029)
      return false;

    // bidirectional text control
    if (CP == 0x200E || CP == 0x200F || (CP >= 0x202A && CP <= 0x202E))
      return false;

    // interlinears and generally specials
    if (CP >= 0xFFF9 && CP <= 0xFFFF)
      return false;

    return true;
  }
}

  // Deals with ascii & utf8 characters being output into stdout
  // As the string is printed, check each character for:
  //  0. Valid printable character
  //  1. Unicode code page
  //  2. Valid format character \t, \n, \r, \f, \v
  //  3. Unknown; data
  // Until case 3 is reached, the string is ouput possibly escaped, but
  // otherwise unadulterated.
  // If case 3 is reached, back up until the last valid printable character
  // ( 0 & 1) and dump all remaining 2& 3 characters as hex.

  class RawStringConverter {
    enum { kBufSize = 1024 };
    enum HexState { kText, kEsc, kHex, kEnd };

    llvm::SmallString<kBufSize> m_Buf;
    std::locale m_Loc;
    bool m_Utf8Out;

    class ByteDumper {
      RawStringConverter& m_Convert;
      const char* const m_End;
      const bool m_Utf8;
      bool m_HexRun;
      bool (*isPrintable)(char32_t, const std::locale&);

      template <class T> static bool isPrint(char32_t C, const std::locale& L) {
        return std::isprint(T(C), L);
      }

    public:
      ByteDumper(RawStringConverter& C, const char* E, bool Utf8)
        : m_Convert(C), m_End(E), m_Utf8(Utf8) {
        // cache the correct isprint variant rather than checking in a loop
        isPrintable = m_Convert.m_Utf8Out && m_Utf8 ? &utf8::isPrint :
                  (m_Utf8 ? &isPrint<wchar_t> : &isPrint<char>);
      }
    
      HexState operator() (const char*& Ptr, llvm::raw_ostream& Stream,
                           bool ForceHex) {
        // Block allocate the next chunk
        if (!(m_Convert.m_Buf.size() % kBufSize))
          m_Convert.m_Buf.reserve(m_Convert.m_Buf.size() + kBufSize);

        HexState State = kText;
        const char* const Start = Ptr;
        char32_t Char;
        if (m_Utf8) {
          Char = utf8::next(Ptr);
          if (Ptr > m_End) {
            // Invalid/bad encoding: dump the remaining as hex
            Ptr = Start;
            while (Ptr < m_End)
              Stream << "\\x" << llvm::format_hex_no_prefix(uint8_t(*Ptr++), 2);
            m_HexRun = true;
            return kHex;
          }
        } else
          Char = (*Ptr++ & 0xff);

        const std::locale& Loc = m_Convert.m_Loc;
        // Assume more often than not -regular- strings are printed
        if (LLVM_UNLIKELY(!isPrintable(Char, Loc))) {
          m_HexRun = false;
          if (LLVM_UNLIKELY(ForceHex || !std::isspace(wchar_t(Char), Loc))) {
            if (Char > 0xffff)
              Stream << "\\U" << llvm::format_hex_no_prefix(uint32_t(Char), 8);
            else if (Char > 0xff)
              Stream << "\\u" << llvm::format_hex_no_prefix(uint16_t(Char), 4);
            else if (Char) {
              Stream << "\\x" << llvm::format_hex_no_prefix(uint8_t(Char), 2);
              m_HexRun = true;
              return kHex;
            } else
              Stream << "\\0";
            return kText;
          }

          switch (Char) {
            case '\b': Stream << "\\b"; return kEsc;
            // \r isn't so great on Unix, what about Windows?
            case '\r': Stream << "\\r"; return kEsc;
            default: break;
          }
          State = kEsc;
        }

        if (m_HexRun) {
          // If the last print was a hex code, and this is now a char that could
          // be interpreted as a continuation of that hex-sequence, close out
          // the string and use concatenation. {'\xea', 'B'} -> "\xea" "B"
          m_HexRun = false;
          if (std::isxdigit(wchar_t(Char), Loc))
            Stream << "\" \"";
        }
        if (m_Utf8)
          Stream << llvm::StringRef(Start, Ptr-Start);
        else
          Stream << char(Char);
        return State;
      }
    };

  public:
    RawStringConverter() : m_Utf8Out(false) {
      if (!::strcasestr(m_Loc.name().c_str(), "utf-8")) {
        if (const char* LANG = ::getenv("LANG")) {
          if (::strcasestr(LANG, "utf-8")) {
            m_Loc = std::locale(LANG);
            m_Utf8Out = true;
          }
        }
      } else
        m_Utf8Out = true;
    }

    llvm::StringRef convert(const char* const Start, size_t N) {
      const char* Ptr = Start;
      const char* const End = Start + N;

      bool isUtf8 = Start[0] != '\"';
      if (!isUtf8) {
        bool isPrint = true;
        // A const char* string may not neccessarily be utf8.
        // When the locale can output utf8 strings, validate it as utf8 first.
        if (!m_Utf8Out) {
          while (isPrint && Ptr < End)
            isPrint = std::isprint(*Ptr++, m_Loc);
        } else
          isUtf8 = utf8::validate(Ptr, N, m_Loc, isPrint);

        // Simple printable string, just return it now.
        if (isPrint)
          return llvm::StringRef(Start, N);

        Ptr = Start;
      } else {
        assert(Start[0] == 'u' || Start[0] == 'U' || Start[0] == 'L'
               && "Unkown string encoding");
        // Unicode string, assume valid
        isUtf8 = true;
      }

      ByteDumper Dump(*this, End, isUtf8);
      { // scope for llvm::raw_svector_ostream
        size_t LastGood = 0;
        HexState Hex = kText;
        llvm::raw_svector_ostream Strm(m_Buf);
        while ((Hex < kEnd) && (Ptr < End)) {
          const size_t LastPos = Ptr - Start;
          switch (Dump(Ptr, Strm, Hex==kHex)) {
            case kHex:
              // Printed hex char
              // Keep doing it as long as we haven't printed an escaped char
              Hex = (Hex == kEsc) ? kEnd : kHex;
              break;
            case kEsc:
              assert(Hex <= kEsc && "Escape code after hex shouldn't occur");
              // Mark this as the last good character printed
              if (Hex == kText)
                LastGood = LastPos;
              Hex = kEsc;
            default:
              break;
          }
        }
        if (Hex != kEnd)
          return Strm.str();

        Ptr = Start + LastGood;
        m_Buf.resize(LastGood);
      }

      llvm::raw_svector_ostream Strm(m_Buf);
      while (Ptr < End)
        Dump(Ptr, Strm, true);
      return Strm.str();
    }
  };

  void Value::dump() const {
    // We need stream that doesn't close its file descriptor, thus we are not
    // using llvm::outs. Keeping file descriptor open we will be able to use
    // the results in pipes (Savannah #99234).
    llvm::raw_os_ostream Out(std::cout);

    // Get the default type string representation
    Out << cling::valuePrinterInternal::printTypeInternal(*this);
    Out << " ";

    // Get the value string representation, by printValue() method overloading
    const std::string Val = cling::valuePrinterInternal::printValueInternal(*this);

    const char* Data = Val.data();
    const size_t N = Val.size();
    switch (N ? Data[0] : 0) {
      case 'u': case 'U': case 'L':
        if (N < 3 || Data[1] != '\"')
          break;
        // Unicode string, encoded as Utf-8
      case '\"':
        if (N > 2 && Data[N-1] == '\"') {
          // Drop the terminating " so Utf-8 errors can be detected ("\xeA")
          Out << RawStringConverter().convert(Data, N-1) << "\"\n";
          return;
        }
      default:
        break;
    }
    Out << Val << '\n';
  }
} // end namespace cling
