//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Roman Zulak
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Output.h"
#include "cling/Utils/UTF8.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"

#ifdef _WIN32
#include <Shlwapi.h>
#define strcasestr StrStrIA
#pragma comment(lib, "Shlwapi.lib")
#endif

namespace cling {
namespace utils {
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

bool Validate(const char* Str, size_t N, const std::locale& LC, bool& isPrint) {
  for (size_t i = 0, N1 = (N-1); i < N; ++i) {
    uint8_t n;
    uint8_t C = mask8(Str[i]);
    isPrint = isPrint ? std::isprint(Str[i], LC) : false;

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

namespace {
  enum HexState {
    kText,
    kEsc,
    kHex,
    kEnd
  };
}

class EscapeSequence::ByteDumper {
  enum { kBufSize = 1024 };
  llvm::SmallString<kBufSize> m_Buf;

  const std::locale& m_Loc;
  const char* const m_End;
  const bool m_Utf8;
  bool m_HexRun;
  bool (* const isPrintable)(char32_t, const std::locale&);

  static bool stdIsPrintU(char32_t C, const std::locale& L) {
#if !defined(_WIN32)
    static_assert(sizeof(wchar_t) == sizeof(char32_t), "Sizes don't match");
    return std::isprint(wchar_t(C), L);
#else
    // Each on their own report a lot of characters printable.
    return iswprint(C) && std::isprint(C, L);
#endif
  }

  static bool stdIsPrintA(char32_t C, const std::locale& L) {
#if defined(_WIN32)
    // Windows Debug does not like being sent values > 255!
    if (C > 0xff)
      return false;
#endif
    return std::isprint(char(C), L);
  }

public:

  ByteDumper(EscapeSequence& Enc, const char* E, bool Utf8) :
    m_Loc(Enc.m_Loc), m_End(E), m_Utf8(Utf8), m_HexRun(false),
    isPrintable(Enc.m_Utf8Out && m_Utf8 ? &utf8::isPrint :
                                         (Utf8 ? &stdIsPrintU : &stdIsPrintA)) {
    // Cached the correct isprint variant rather than checking in a loop.
    //
    // If output handles UTF-8 and string is UTF-8, then validate against UTF-8.
    // If string is UTF-8 validate printable against std::isprint<char32_t>.
    // If nothing is UTF-8 validate against std::isprint<char> .
  }

  HexState operator() (const char*& Ptr, llvm::raw_ostream& Stream,
                       bool ForceHex) {
    // Block allocate the next chunk
    if (!(m_Buf.size() % kBufSize))
      m_Buf.reserve(m_Buf.size() + kBufSize);

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

    // Assume more often than not -regular- strings are printed
    if (LLVM_UNLIKELY(!isPrintable(Char, m_Loc))) {
      m_HexRun = false;
      if (LLVM_UNLIKELY(ForceHex || !std::isspace(wchar_t(Char), m_Loc))) {
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
      if (std::isxdigit(wchar_t(Char), m_Loc))
        Stream << "\" \"";
    }
    if (m_Utf8)
      Stream << llvm::StringRef(Start, Ptr-Start);
    else
      Stream << char(Char);
    return State;
  }
  llvm::SmallString<kBufSize>& buf() { return m_Buf; }
};

EscapeSequence::EscapeSequence() : m_Utf8Out(false) {
#if !defined(_WIN32)
  if (!::strcasestr(m_Loc.name().c_str(), "utf-8")) {
    if (const char* LANG = ::getenv("LANG")) {
      if (::strcasestr(LANG, "utf-8")) {
 #if !defined(__APPLE__) || !defined(__GLIBCXX__)
        m_Loc = std::locale(LANG);
 #endif
        m_Utf8Out = true;
      }
    }
  } else
    m_Utf8Out = true;
#else
  // Can other other codepages support UTF-8?
  m_Utf8Out = ::GetConsoleOutputCP() == 65001;
#endif
}

llvm::raw_ostream& EscapeSequence::encode(const char* const Start, size_t N,
                                          llvm::raw_ostream& Output) {
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
      isUtf8 = Validate(Ptr, N, m_Loc, isPrint);

    // Simple printable string, just return it now.
    if (isPrint)
      return Output << llvm::StringRef(Start, N);

    Ptr = Start;
  } else {
    assert((Start[0] == 'u' || Start[0] == 'U' || Start[0] == 'L')
           && "Unkown string encoding");
    // Unicode string, assume valid
    isUtf8 = true;
  }

  ByteDumper Dump(*this, End, isUtf8);
  { // scope for llvm::raw_svector_ostream
    size_t LastGood = 0;
    HexState Hex = kText;
    llvm::raw_svector_ostream Strm(Dump.buf());
    while ((Hex < kEnd) && (Ptr < End)) {
      const size_t LastPos = Ptr - Start;
      switch (Dump(Ptr, Strm, Hex==kHex)) {
        case kHex:
          // Printed hex char
          // Keep doing it as long as no escaped char have been printed
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
      return Output << Strm.str();

    Ptr = Start + LastGood;
    Dump.buf().resize(LastGood);
  }

  // Force hex output for the rest of the string
  llvm::raw_svector_ostream Strm(Dump.buf());
  while (Ptr < End)
    Dump(Ptr, Strm, true);

  return Output << Strm.str();
}

std::string EscapeSequence::encode(const char* const Start, size_t N) {
  stdstrstream Strm;
  encode(Start, N, Strm);
  return Strm.str();
}

} // namespace cling
} // namespace utils
} // namespace utf8
