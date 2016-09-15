//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Platform.h"

namespace cling {
  namespace utils {
    // Checking whether the pointer points to a valid memory location
    bool isAddressValid(const void *P) {
      if (!P || P == (void *) -1)
        return false;

      return platform::IsMemoryValid(P);
    }
  }
}
