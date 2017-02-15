//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author: Roman Zulak
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_UTILS_UTF8_H
#define CLING_UTILS_UTF8_H

#include "llvm/Support/raw_ostream.h"
#include <locale>
#include <string>

namespace cling {

  ///\brief Encode a series of characters as UTF-8.
  ///
  /// \param [in] Str - Pointer to the first chracter to encode
  /// \param [in] N - Number of characters to encode
  /// \param [in] Prefix - String quoting & literal prefix to use:
  ///             0 - No prefix, raw encoding: toUTF8("STR", 3) => STR (len 3)
  ///             1 - Quoted string: toUTF8("STR", 3) => "STR" (len 5)
  ///             character: Quoted string prefixed by the character
  ///             toUTF8("STR", 3, 'u') => u"STR" (len 6)
  ///
  /// \return The UTF-8 encoded string
  ///
  template <typename T>
  std::string toUTF8(const T* const Str, size_t N, const char Prefix = 0);

  namespace utils {
    namespace utf8 {

      ///\brief Validate a seried of bytes as properly encoded UTF-8
      ///
      /// \param [in] Str - Pointer to the first byte to validate
      /// \param [in] N - Number of bytes to validate
      /// \param [in] Loc: std::locale to test if Str is also printable
      /// \param [out] IsPrint - Whether all of the characters are printable
      ///
      /// \return true if Str to Str+N is a valid UTF-8 run.
      ///
      bool Validate(const char* Str, size_t N, const std::locale& Loc,
                    bool& IsPrint);

      ///\brief EscapeSequence encodes a series of bytes into a version suitable
      // for printing in the current locale, or serialization.
      // As the string is printed, check each character for:
      //  0. Valid printable character
      //  1. Unicode code page
      //  2. Valid format character \t, \n, \r, \f, \v
      //  3. Unknown; data
      // Until case 3 is reached, the string is ouput possibly escaped, but
      // otherwise unadulterated.
      // If case 3 is reached, back up until the last valid printable character
      // ( 0 & 1) and dump all remaining 2 & 3 characters as hex.

      class EscapeSequence {
        class ByteDumper;
        std::locale m_Loc;
        bool m_Utf8Out;

      public:
        EscapeSequence();

        ///\brief Encode the bytes from Str into a representation capable of
        /// being printed in the current locale without data loss.
        /// When Str begins with any of the C++ unicode string literal
        /// (u", U", L", u8"), it is assumed to be a valid UTF-8 string.
        ///
        /// \param [in] Str - Start of bytes to convert
        /// \param [in] N - Number of bytes to convert
        /// \param [in] Output - Ouput stream to write to
        ///
        /// \return 'Output' to allow: EscapeSequence().encode(...) << "";
        ///
        llvm::raw_ostream& encode(const char* const Str, size_t N,
                                  llvm::raw_ostream& Output);

        ///\brief Overload for above returning a std::string.
        ///
        std::string encode(const char* const Str, size_t N);
      };
    }
  }
}

#endif // CLING_UTILS_UTF8_H
