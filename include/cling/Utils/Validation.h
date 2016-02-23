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

#include "llvm/Config/config.h" // for LLVM_ON_WIN32

#include <assert.h>
#include <errno.h>
#ifdef LLVM_ON_WIN32
# define WIN32_LEAN_AND_MEAN
# define NOGDI
# include <Windows.h>
#else
# include <unistd.h>
#endif

namespace cling {
  namespace utils{
    // Checking whether the pointer points to a valid memory location
    // Used for checking of void* output
    // Should be moved to earlier stages (ex. IR) in the future
    static bool isAddressValid(const void *P) {
      if (!P || P == (void *) -1)
        return false;

#ifdef LLVM_ON_WIN32
      MEMORY_BASIC_INFORMATION MBI;
      if (!VirtualQuery(P, &MBI, sizeof(MBI)))
        return false;
      if (MBI.State != MEM_COMMIT)
        return false;
      return true;
#else
      // There is a POSIX way of finding whether an address can be accessed for
      // reading: write() will return EFAULT if not.
      int FD[2];
      if (pipe(FD))
        return false; // error in pipe()? Be conservative...
      int NBytes = write(FD[1], P, 1/*byte*/);
      close(FD[0]);
      close(FD[1]);
      if (NBytes != 1) {
        assert(errno == EFAULT && "unexpected pipe write error");
        return false;
      }
      return true;
#endif
    }
  }
}

#endif // CLING_UTILS_VALIDATION_H
