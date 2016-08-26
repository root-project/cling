//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_UTILS_SOURCE_NORMALIZATION_H
#define CLING_UTILS_SOURCE_NORMALIZATION_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace clang {
  class LangOptions;
  class SourceLocation;
  class SourceManager;
}

namespace cling {
namespace utils {
  ///\brief Determine whether the source is an unnamed macro.
  ///
  /// Unnamed macros contain no function definition, but "prompt-style" code
  /// surrounded by a set of curly braces.
  ///
  /// \param source The source code to analyze.
  /// \param LangOpts - LangOptions to use for lexing.
  /// \return the position of the unnamed macro's opening '{'; or
  ///         std::string::npos if this is not an unnamed macro.
  size_t isUnnamedMacro(llvm::StringRef source,
                        clang::LangOptions& LangOpts);

  ///\brief Determine whether the source needs to be moved into a function.
  ///
  /// If so, move possible includes directives out of the future body of the
  /// function and return the position where the function signature should be
  /// inserted.
  ///
  /// \param source - The source code to analyze; out: the source with
  ///        re-arranged includes.
  /// \param LangOpts - LangOptions to use for lexing.
  /// \return The position where the function signature and '{' should be
  ///     inserted; std::string::npos if this source should not be wrapped.
  size_t getWrapPoint(std::string& source, const clang::LangOptions& LangOpts);
} // namespace utils
} // namespace cling

#endif // CLING_UTILS_SOURCE_NORMALIZATION_H
