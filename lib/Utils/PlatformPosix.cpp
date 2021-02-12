//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author: Roman Zulak
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Platform.h"

#if defined(LLVM_ON_UNIX)

#include "cling/Utils/Paths.h"
#include "llvm/ADT/SmallString.h"

#include <array>
#include <atomic>
#include <string>
#include <cxxabi.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// PATH_MAX
#ifdef __APPLE__
 #include <sys/syslimits.h>
#else
 #include <limits.h>
#endif

#define PATH_MAXC (PATH_MAX+1)

namespace cling {
namespace utils {
namespace platform {

namespace {
  struct PointerCheck {
  private:
    // A simple round-robin cache: what enters first, leaves first.
    // MRU cache wasn't worth the extra CPU cycles.
    static thread_local std::array<const void*, 8> lines;
    static thread_local unsigned mostRecent;
    size_t page_size;
    size_t page_mask;

    // Concurrent writes to the same cache element can result in invalid cache
    // elements, causing pointer address not being available in the cache even
    // though they should be, i.e. false cache misses. While can cause a
    // slow-down, the cost for keeping the cache thread-local or atomic is
    // much higher (yes, this was measured).
    void push(const void* P) {
      mostRecent = (mostRecent + 1) % lines.size();
      lines[mostRecent] = P;
    }

  public:
    PointerCheck() : page_size(::sysconf(_SC_PAGESIZE)), page_mask(~(page_size - 1))
    {
       assert(IsPowerOfTwo(page_size));
    }

    bool operator () (const void* P) {
      // std::find is considerably slower, do manual search instead.
      if (P == lines[0] || P == lines[1] || P == lines[2] || P == lines[3]
          || P == lines[4] || P == lines[5] || P == lines[6] || P == lines[7])
        return true;

      // Address of page containing P, assuming page_size is a power of 2
      void *base = (void *)(((size_t)P) & page_mask);

      // P is invalid only when msync returns -1 and sets errno to ENOMEM
      if (::msync(base, page_size, MS_ASYNC) != 0) {
        assert(errno == ENOMEM && "Unexpected error in call to msync()");
        return false;
      }

      push(P);
      return true;
    }
  private:
    bool IsPowerOfTwo(size_t n) {
       /* While n is even and larger than 1, divide by 2 */
       while (((n & 1) == 0) && n > 1)
         n >>= 1;
       return n == 1;
    }
  };
  thread_local std::array<const void*, 8> PointerCheck::lines = {};
  thread_local unsigned PointerCheck::mostRecent = 0;
}

bool IsMemoryValid(const void *P) {
  static PointerCheck sPointerCheck;
  return sPointerCheck(P);
}

std::string GetCwd() {
  char Buffer[PATH_MAXC];
  if (::getcwd(Buffer, sizeof(Buffer)))
    return Buffer;

  ::perror("Could not get current working directory");
  return std::string();
}

static void DLErr(std::string* Err) {
  if (!Err)
    return;
  if (const char* DyLibError = ::dlerror())
    *Err = DyLibError;
}

const void* DLOpen(const std::string& Path, std::string* Err) {
  void* Lib = dlopen(Path.c_str(), RTLD_LAZY|RTLD_GLOBAL);
  DLErr(Err);
  return Lib;
}

const void* DLSym(const std::string& Name, std::string* Err) {
  if (const void* Self = ::dlopen(nullptr, RTLD_GLOBAL)) {
    // get dlopen error if there is one
    DLErr(Err);
    const void* Sym = ::dlsym(const_cast<void*>(Self), Name.c_str());
    // overwrite error if dlsym caused one
    DLErr(Err);
    // only get dlclose error if dlopen & dlsym haven't emited one
    DLClose(Self, Err && Err->empty() ? Err : nullptr);
    return Sym;
  }
  DLErr(Err);
  return nullptr;
}

void DLClose(const void* Lib, std::string* Err) {
  ::dlclose(const_cast<void*>(Lib));
  DLErr(Err);
}

std::string NormalizePath(const std::string& Path) {
  char Buf[PATH_MAXC];
  if (const char* Result = ::realpath(Path.c_str(), Buf))
    return std::string(Result);

  ::perror("realpath");
  return std::string();
}

bool Popen(const std::string& Cmd, llvm::SmallVectorImpl<char>& Buf, bool RdE) {
  if (FILE *PF = ::popen(RdE ? (Cmd + " 2>&1").c_str() : Cmd.c_str(), "r")) {
    Buf.resize(0);
    const size_t Chunk = Buf.capacity_in_bytes();
    while (true) {
      const size_t Len = Buf.size();
      Buf.resize(Len + Chunk);
      const size_t R = ::fread(&Buf[Len], sizeof(char), Chunk, PF);
      if (R < Chunk) {
        Buf.resize(Len + R);
        break;
      }
    }
    ::pclose(PF);
    return !Buf.empty();
  }
  return false;
}

bool GetSystemLibraryPaths(llvm::SmallVectorImpl<std::string>& Paths) {
#if defined(__APPLE__) || defined(__CYGWIN__)
  Paths.push_back("/usr/local/lib/");
  Paths.push_back("/usr/X11R6/lib/");
  Paths.push_back("/usr/lib/");
  Paths.push_back("/lib/");

 #ifndef __APPLE__
  Paths.push_back("/lib/x86_64-linux-gnu/");
  Paths.push_back("/usr/local/lib64/");
  Paths.push_back("/usr/lib64/");
  Paths.push_back("/lib64/");
 #endif
#else
  llvm::SmallString<1024> Buf;
  platform::Popen("LD_DEBUG=libs LD_PRELOAD=DOESNOTEXIST ls", Buf, true);
  const llvm::StringRef Result = Buf.str();

  const std::size_t NPos = std::string::npos;
  const std::size_t LD = Result.find("(LD_LIBRARY_PATH)");
  std::size_t From = Result.find("search path=", LD == NPos ? 0 : LD);
  if (From != NPos) {
    const std::size_t To = Result.find("(system search path)", From);
    if (To != NPos) {
      From += 12;
      std::string SysPath = Result.substr(From, To-From);
      SysPath.erase(std::remove_if(SysPath.begin(), SysPath.end(), isspace),
                    SysPath.end());

      llvm::SmallVector<llvm::StringRef, 10> CurPaths;
      SplitPaths(SysPath, CurPaths);
      for (const auto& Path : CurPaths)
        Paths.push_back(Path.str());
    }
  }
#endif
  return true;
}

std::string Demangle(const std::string& Symbol) {
  struct AutoFree {
    char* Str;
    AutoFree(char* Ptr) : Str(Ptr) {}
    ~AutoFree() { ::free(Str); };
  };
  int status = 0;
  // Some implementations of __cxa_demangle are giving back length of allocation
  // Passing NULL for length seems to guarantee null termination.
  AutoFree af(abi::__cxa_demangle(Symbol.c_str(), NULL, NULL, &status));
  return status == 0 ? std::string(af.Str) : std::string();
}

} // namespace platform
} // namespace utils
} // namespace cling

#endif // LLVM_ON_UNIX
