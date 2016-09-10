//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_OUTPUT_H
#define CLING_OUTPUT_H

#include "llvm/Support/raw_ostream.h"

namespace cling {
  namespace utils {
    ///\brief The 'stdout' stream. llvm::raw_ostream wrapper of std::cout
    ///
    llvm::raw_ostream& outs();

    ///\brief The 'stderr' stream. llvm::raw_ostream wrapper of std::cerr
    ///
    llvm::raw_ostream& errs();

    ///\brief The 'logging' stream. Currently returns cling::errs().
    /// This matches clang & gcc prinitng to stderr for certain information.
    /// If the host process needs to keep stderr for itself or actual errors,
    /// then the function can be edited to return a separate stream.
    ///
    llvm::raw_ostream& log();
  }
  using utils::outs;
  using utils::errs;
  using utils::log;
}

#endif // CLING_OUTPUT_H
