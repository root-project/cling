//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Output.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_os_ostream.h"

#include <iostream>

namespace cling {
  namespace utils {
    llvm::raw_ostream& outs() {
      static llvm::raw_os_ostream sOut(std::cout);
      sOut.SetUnbuffered();
      if (llvm::sys::Process::StandardOutIsDisplayed())
        sOut.enable_colors(true);
      return sOut;
    }

    llvm::raw_ostream& errs() {
      static llvm::raw_os_ostream sErr(std::cerr);
      sErr.SetUnbuffered();
      if (llvm::sys::Process::StandardErrIsDisplayed())
        sErr.enable_colors(true);
      return sErr;
    }

    llvm::raw_ostream& log() {
      return cling::errs();
    }
  }
}
