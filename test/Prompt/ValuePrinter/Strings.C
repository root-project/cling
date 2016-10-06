//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

//RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

#include <string>

std::string(u8"UTF-8")
// CHECK: (std::string) "UTF-8"

std::u16string(u"UTF-16 " u"\x394" u"\x3a6" u"\x3a9")
// CHECK-NEXT: (std::u16string) u"UTF-16 ΔΦΩ"

std::u32string(U"UTF-32 " U"\x262D" U"\x2615" U"\x265F")
// CHECK-NEXT: (std::u32string) U"UTF-32 ☭☕♟"

std::wstring(L"wide")
// CHECK-NEXT: (std::wstring) L"wide"

u"16strliteral"
// CHECK-NEXT: (const char16_t [13]) u"16strliteral"

U"32literalstr"
// CHECK-NEXT: (const char32_t [13]) U"32literalstr"

L"wcharliteral"
// CHECK-NEXT: (const wchar_t [13]) L"wcharliteral"

// Unicode shouldn't do character level access, return the raw codepages

const char16_t* utf16 = u"16str";
utf16[0]
// CHECK-NEXT: (const char16_t) u'\u0031'

const char32_t* utf32 = U"32str";
utf32[1]
// CHECK-NEXT: (const char32_t) U'\U00000032'

// wchar_t doesn't guarantee a size
const wchar_t* wides = L"wide";
wides[3]
// CHECK-NEXT: (const wchar_t) L'\x{{0+}}65'

// expected-no-diagnostics
.q
