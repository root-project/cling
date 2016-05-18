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
#include "llvm/Config/llvm-config.h"

#include <assert.h>
#include <errno.h>
#ifdef LLVM_ON_WIN32
# include <Windows.h>
#else
# include <unistd.h>
#endif
#include <array>
#include <algorithm>
#include <fcntl.h>
#include <atomic>

namespace cling {
  namespace utils {

#ifndef LLVM_ON_WIN32
    // A simple round-robin cache: what enters first, leaves first.
    // MRU cache wasn't worth the extra CPU cycles.
    struct Cache {
    private:
      std::array<const void*, 8> lines;
      std::atomic<unsigned> mostRecent = {0};
    public:
      bool contains(const void* P) {
        return std::find(lines.begin(), lines.end(), P) != lines.end();
      }

      // Concurrent writes to the same cache element can result in invalid cache
      // elements, causing pointer address not being available in the cache even
      // though they should be, i.e. false cache misses. While can cause a
      // slow-down, the cost for keeping the cache thread-local or atomic is
      // much higher (yes, this was measured).
      void push(const void* P) {
        unsigned acquiredVal = mostRecent;
        while(!mostRecent.compare_exchange_weak(acquiredVal, (acquiredVal+1)%lines.size())) {
          acquiredVal = mostRecent;
        }
        lines[acquiredVal] = P;
      }
    };

    // Note: not thread safe, see comment above push().
    static Cache& getCache() {
      static Cache threadCache;
      return threadCache;
    }

    static int getNullDevFileDescriptor() {
      struct FileDescriptor {
        int FD;
        const char* file = "/dev/random";
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
      if (currentCache.contains(P))
        return true;
      // There is a POSIX way of finding whether an address
      // can be accessed for reading.
      if (write(getNullDevFileDescriptor(), P, 1/*byte*/) != 1) {
        assert(errno == EFAULT && "unexpected write error at address");
        return false;
      }
      currentCache.push(P);
      return true;
#endif
    }
  }
}
