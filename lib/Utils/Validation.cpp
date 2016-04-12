//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Validation.h"
#include "llvm/Support/ThreadLocal.h"

#include <assert.h>
#include <errno.h>
#ifdef LLVM_ON_WIN32
# include <Windows.h>
#else
#include <fcntl.h>
# include <unistd.h>
#endif

#define CACHESIZE 8

namespace cling {
  namespace utils {

#ifndef LLVM_ON_WIN32
    struct Cache {
    private:
      const void* lines[CACHESIZE];
      unsigned size, mostRecent;
    public:
      Cache(): lines{0,0,0,0,0,0,0,0}, size(CACHESIZE), mostRecent(0) {}
      bool findInCache(const void* P) {
        for (unsigned index = 0; index < size; index++) {
          if (lines[index] == P)
            return true;
        }
        return false;
      }
      void pushToCache(const void* P) {
        mostRecent = (mostRecent+1)%size;
        lines[mostRecent] = P;
      }
    };

    // Trying to be thread-safe.
    // Each thread creates a new cache when needed.
    static Cache& getCache() {
      static llvm::sys::ThreadLocal<Cache> threadCache;
      if (!threadCache.get()) {
        threadCache.set(new Cache());
      }
      return *threadCache.get();
    }

    static int getNullDevFileDescriptor() {
      struct FileDescriptor {
        int FD;
        const char* file = "/dev/null";
        FileDescriptor() { FD = open(file, O_WRONLY); }
        ~FileDescriptor() {
          close(FD);
        }
      };
      static FileDescriptor nullDev;
      return nullDev.FD;
    }
#endif

    // Checking whether the pointer points to a valid memory location
    bool isAddressValid(const void *P) {
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
      // Look-up the address in the cache.
      Cache& currentCache = getCache();
      if (currentCache.findInCache(P))
        return true;
      // There is a POSIX way of finding whether an address
      // can be accessed for reading.
      if (write(getNullDevFileDescriptor(), P, 1/*byte*/) != 1) {
        assert(errno == EFAULT && "unexpected write error at address");
        return false;
      }
      currentCache.pushToCache(P);
      return true;
#endif
    }
  }
}
