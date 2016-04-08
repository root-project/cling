//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_UTILS_VALIDATION_H
#define CLING_UTILS_VALIDATION_H

namespace cling {
  namespace utils{
    // Checking whether the pointer points to a valid memory location
    // Used for checking of void* output
    // Should be moved to earlier stages (ex. IR) in the future
    bool isAddressValid(const void *P);
  }
}

#endif // CLING_UTILS_VALIDATION_H
