//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author: Roman Zulak
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Platform.h"

#ifdef __APPLE__

#include "cling/Utils/Paths.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <mach/vm_map.h>
#include <mach/mach_host.h>

#include <CoreFoundation/CFBase.h> // For MAC_OS_X_VERSION_X_X macros

// gcc on Mac can only include CoreServices.h up to 10.9 SDK, which means
// we cannot use Gestalt to get the running OS version when >= 10.10
#if defined(__clang__) || !defined(MAC_OS_X_VERSION_10_10)
#include <dlfcn.h> // dlopen to avoid linking with CoreServices
#include <CoreServices/CoreServices.h>
#else
#define CLING_SWVERS_PARSE_ONLY 1
#endif

namespace cling {
namespace utils {
namespace platform {

bool IsMemoryValid(const void *P) {
  char Buf;
  vm_size_t sizeRead = sizeof(Buf);
  if (::vm_read_overwrite(mach_task_self(), vm_address_t(P), sizeRead,
                          vm_address_t(&Buf), &sizeRead) != KERN_SUCCESS)
    return false;
  if (sizeRead != sizeof(Buf))
    return false;

  return true;
}

inline namespace osx {

namespace {

static bool getISysRootVersion(const std::string& SDKs, int Major,
                               int Minor, std::string& SysRoot,
                               const char* Verbose) {
  std::ostringstream os;
  os << SDKs << "MacOSX" << Major << "." << Minor << ".sdk";

  std::string SDKv = os.str();
  if (llvm::sys::fs::is_directory(SDKv)) {
    SysRoot.swap(SDKv);
    if (Verbose) {
      llvm::errs() << "SDK version matching " << Major << "." << Minor
                   << " found, this does " << Verbose << "\n";
    }
    return true;
  }

  if (Verbose)
    llvm::errs() << "SDK version matching " << Major << "." << Minor
                 << " not found, this would " << Verbose << "\n";

  return false;
}

static std::string ReadSingleLine(const char* Cmd) {
  if (FILE* PF = ::popen(Cmd, "r")) {
    char Buf[1024];
    char* BufPtr = ::fgets(Buf, sizeof(Buf), PF);
    ::pclose(PF);
    if (BufPtr && Buf[0]) {
      const llvm::StringRef Result(Buf);
      assert(Result[Result.size()-1] == '\n' && "Single line too large");
      return Result.trim().str();
    }
  }
  return "";
}

} // anonymous namespace

bool GetISysRoot(std::string& sysRoot, bool Verbose) {
  using namespace llvm::sys;

  // Some versions of OS X and Server have headers installed
  if (fs::is_regular_file("/usr/include/stdlib.h"))
    return false;

  std::string SDKs("/Applications/Xcode.app/Contents/Developer");

  // Is XCode installed where it usually is?
  if (!fs::is_directory(SDKs)) {
    // Nope, use xcode-select -p to get the path
    SDKs = ReadSingleLine("xcode-select -p");
    if (SDKs.empty())
      return false;  // Nothing more we can do
  }

  SDKs.append("/Platforms/MacOSX.platform/Developer/SDKs/");
  if (!fs::is_directory(SDKs))
    return false;


  // Try to get the SDK for whatever version of OS X is currently running
  // Seems to make more sense to get the currently running SDK so headers
  // and any loaded libraries will match.

  int32_t majorVers = -1, minorVers = -1;
#ifndef CLING_SWVERS_PARSE_ONLY
 #pragma clang diagnostic push
 #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  if (void *core = ::dlopen(
          "/System/Library/Frameworks/CoreServices.framework/CoreServices",
          RTLD_LAZY)) {
    typedef ::OSErr (*GestaltProc)(::OSType, ::SInt32 *);
    if (GestaltProc Gestalt = (GestaltProc)dlsym(core, "Gestalt")) {
      if (Gestalt(gestaltSystemVersionMajor, &majorVers) == ::noErr) {
        if (Gestalt(gestaltSystemVersionMinor, &minorVers) != ::noErr)
          minorVers = -1;
      } else
        majorVers = -1;
    }
    ::dlclose(core);
  }
 #pragma clang diagnostic pop
#endif

  if (majorVers == -1 || minorVers == -1) {
    const std::string SWVers = ReadSingleLine("sw_vers | grep ProductVersion"
                                              " | awk '{print $2}'");
    if (!SWVers.empty()) {
      if (::sscanf(SWVers.c_str(), "%d.%d", &majorVers, &minorVers) != 2) {
        majorVers = -1;
        minorVers = -1;
      }
    }
  }

  if (majorVers != -1 && minorVers != -1) {
    if (getISysRootVersion(SDKs, majorVers, minorVers, sysRoot,
            Verbose ? "match the version of OS X running"
                    : nullptr)) {
      return true;
    }
  }


#define GET_ISYSROOT_VER(maj, min) \
  if (getISysRootVersion(SDKs, maj, min, sysRoot, Verbose ? \
                 "match what cling was compiled with" : nullptr)) \
    return true;

  // Try to get the SDK for whatever cling was compiled with
  #if defined(MAC_OS_X_VERSION_10_11)
    GET_ISYSROOT_VER(10, 11);
  #elif defined(MAC_OS_X_VERSION_10_10)
    GET_ISYSROOT_VER(10, 10);
  #elif defined(MAC_OS_X_VERSION_10_9)
    GET_ISYSROOT_VER(10, 9);
  #elif defined(MAC_OS_X_VERSION_10_8)
    GET_ISYSROOT_VER(10, 8);
  #elif defined(MAC_OS_X_VERSION_10_7)
    GET_ISYSROOT_VER(10, 7);
  #elif defined(MAC_OS_X_VERSION_10_6)
    GET_ISYSROOT_VER(10, 6);
  #elif defined(MAC_OS_X_VERSION_10_5)
    GET_ISYSROOT_VER(10, 5);
  #elif defined(MAC_OS_X_VERSION_10_4)
    GET_ISYSROOT_VER(10, 4);
  #elif defined(MAC_OS_X_VERSION_10_3)
    GET_ISYSROOT_VER(10, 3);
  #elif defined(MAC_OS_X_VERSION_10_2)
    GET_ISYSROOT_VER(10, 2);
  #elif defined(MAC_OS_X_VERSION_10_1)
    GET_ISYSROOT_VER(10, 1);
  #else // MAC_OS_X_VERSION_10_0
    GET_ISYSROOT_VER(10, 0);
  #endif

#undef GET_ISYSROOT_VER

  // Nothing left to do but iterate the SDKs directory
  // copy the paths and then sort for the latest
  std::error_code ec;
  std::vector<std::string> srtd;
  for (fs::directory_iterator it(SDKs, ec), e; it != e; it.increment(ec))
    srtd.push_back(it->path());
  if (!srtd.empty()) {
    std::sort(srtd.begin(), srtd.end(), std::greater<std::string>());
    sysRoot.swap(srtd[0]);
    return true;
  }

  return false;
}


} // namespace osx
} // namespace platform
} // namespace utils
} // namespace cling

#endif // __APPLE__
