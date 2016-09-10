//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Output.h"
#include "llvm/Support/raw_os_ostream.h"
#include <atomic>
#include <iostream>

namespace cling {
  namespace utils {

    ///\brief Unbuffer the llvm::raw_os_ostream wrapper
    /// Shouldn't affect the underlying iostream buffering
    template <unsigned I>
    static inline void Unbuffer(llvm::raw_os_ostream& S) {
      static std::atomic<unsigned char> Unbufferd(0);
      // Prevent overflow
      if (!Unbufferd) {
        // Run exactly once
        if (++Unbufferd == 1)
          S.SetUnbuffered();
      }
    }

    
    llvm::raw_ostream& outs() {
      // We need stream that doesn't close its file descriptor, thus we are not
      // using llvm::outs. Keeping file descriptor open we will be able to use
      // the results in pipes (Savannah #99234).

      static llvm::raw_os_ostream S(std::cout);
      Unbuffer<1>(S);
      return S;
    }

    llvm::raw_ostream& errs() {
      static llvm::raw_os_ostream S(std::cerr);
      // For command line applications, it make sense to run unbuffered
      // so that we will synch with llvm::errs()
      Unbuffer<2>(S);
      return S;
    }

    llvm::raw_ostream& log() {
      return cling::errs();
    }
  }
}
