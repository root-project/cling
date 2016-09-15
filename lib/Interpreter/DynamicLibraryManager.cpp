//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/DynamicLibraryManager.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/InvocationOptions.h"
#include "cling/Utils/Paths.h"
#include "cling/Utils/Platform.h"

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#ifdef LLVM_ON_WIN32
#include <Windows.h>
#else
#ifdef __APPLE__
 #include <sys/syslimits.h> // PATH_MAX
#else
 #include <limits.h> // PATH_MAX
#endif
#include <dlfcn.h>
#endif

namespace cling {
  DynamicLibraryManager::DynamicLibraryManager(const InvocationOptions& Opts)
    : m_Opts(Opts), m_Callbacks(0) {
    const llvm::SmallVector<const char*, 10> kSysLibraryEnv = {
      "LD_LIBRARY_PATH",
  #if __APPLE__
      "DYLD_LIBRARY_PATH",
      "DYLD_FALLBACK_LIBRARY_PATH",
      /*
      "DYLD_VERSIONED_LIBRARY_PATH",
      "DYLD_FRAMEWORK_PATH",
      "DYLD_FALLBACK_FRAMEWORK_PATH",
      "DYLD_VERSIONED_FRAMEWORK_PATH",
      */
  #elif defined(LLVM_ON_WIN32)
      "PATH",
  #endif
    };
  #if defined(LLVM_ON_WIN32)
    const llvm::StringRef Delim(";");
  #else
    const llvm::StringRef Delim(":");
  #endif

    // Behaviour is to not add paths that don't exist...In an interpreted env
    // does this make sense? Path could pop into existance at any time.
    for (const char* Var : kSysLibraryEnv) {
      if (const char* Env = ::getenv(Var)) {
        llvm::SmallVector<llvm::StringRef, 10> CurPaths;
        SplitPaths(Env, CurPaths, utils::kPruneNonExistant, Delim);
        for (const auto& Path : CurPaths)
          m_SystemSearchPaths.push_back(Path.str());
      }
    }

    platform::GetSystemLibraryPaths(m_SystemSearchPaths);
    m_SystemSearchPaths.push_back(".");
  }

  DynamicLibraryManager::~DynamicLibraryManager() {}

  static bool isSharedLib(llvm::StringRef LibName, bool* exists = 0) {
    using namespace llvm::sys::fs;
    file_magic Magic;
    const std::error_code Error = identify_magic(LibName, Magic);
    if (exists)
      *exists = !Error;

    return !Error &&
#ifdef __APPLE__
      (Magic == file_magic::macho_fixed_virtual_memory_shared_lib
       || Magic == file_magic::macho_dynamically_linked_shared_lib
       || Magic == file_magic::macho_dynamically_linked_shared_lib_stub
       || Magic == file_magic::macho_universal_binary)
#elif defined(LLVM_ON_UNIX)
#ifdef __CYGWIN__
      (Magic == file_magic::pecoff_executable)
#else
      (Magic == file_magic::elf_shared_object)
#endif
#elif defined(LLVM_ON_WIN32)
      (Magic == file_magic::pecoff_executable || platform::IsDLL(LibName.str()))
#else
# error "Unsupported platform."
#endif
      ;
  }

  std::string
  DynamicLibraryManager::lookupLibInPaths(llvm::StringRef libStem) const {
    llvm::SmallVector<std::string, 128>
      Paths(m_Opts.LibSearchPath.begin(), m_Opts.LibSearchPath.end());
    Paths.append(m_SystemSearchPaths.begin(), m_SystemSearchPaths.end());

    for (llvm::SmallVectorImpl<std::string>::const_iterator
           IPath = Paths.begin(), E = Paths.end();IPath != E; ++IPath) {
      llvm::SmallString<512> ThisPath(*IPath); // FIXME: move alloc outside loop
      llvm::sys::path::append(ThisPath, libStem);
      bool exists;
      if (isSharedLib(ThisPath.str(), &exists))
        return ThisPath.str();
      if (exists)
        return "";
    }
    return "";
  }

  std::string
  DynamicLibraryManager::lookupLibMaybeAddExt(llvm::StringRef libStem) const {
    using namespace llvm::sys;

    std::string foundDyLib = lookupLibInPaths(libStem);

    if (foundDyLib.empty()) {
      // Add DyLib extension:
      llvm::SmallString<512> filenameWithExt(libStem);
#if defined(LLVM_ON_UNIX)
#ifdef __APPLE__
      llvm::SmallString<512>::iterator IStemEnd = filenameWithExt.end() - 1;
#endif
      static const char* DyLibExt = ".so";
#elif defined(LLVM_ON_WIN32)
      static const char* DyLibExt = ".dll";
#else
# error "Unsupported platform."
#endif
      filenameWithExt += DyLibExt;
      foundDyLib = lookupLibInPaths(filenameWithExt);
#ifdef __APPLE__
      if (foundDyLib.empty()) {
        filenameWithExt.erase(IStemEnd + 1, filenameWithExt.end());
        filenameWithExt += ".dylib";
        foundDyLib = lookupLibInPaths(filenameWithExt);
      }
#endif
    }

    if (foundDyLib.empty())
      return "";

    // get canonical path name and check if already loaded
#if defined(LLVM_ON_WIN32)
    llvm::SmallString<_MAX_PATH> FullPath("");
    char *res = _fullpath((char *)FullPath.data(), foundDyLib.c_str(), _MAX_PATH);
#else
    llvm::SmallString<PATH_MAX+1> FullPath("");
    char *res = realpath(foundDyLib.c_str(), (char *)FullPath.data());
#endif
    if (res == 0) {
      llvm::errs() << "cling::DynamicLibraryManager::lookupLibMaybeAddExt(): "
        "error getting real (canonical) path of library " << foundDyLib << '\n';
      return foundDyLib;
    }
    FullPath.set_size(strlen(res));
    return FullPath.str();
  }

  std::string DynamicLibraryManager::normalizePath(llvm::StringRef path) {
    // Make the path canonical if the file exists.
    struct stat buffer;
    if (stat(path.data(), &buffer) != 0)
      return "";
#if defined(LLVM_ON_WIN32)
    char buf[_MAX_PATH];
    char *res = _fullpath(buf, path.data(), _MAX_PATH);
#else
    char buf[PATH_MAX+1];
    char *res = realpath(path.data(), buf);
#endif
    if (res == 0) {
      assert(0 && "Cannot normalize!?");
      return "";
    }
    return res;
  }

  std::string
  DynamicLibraryManager::lookupLibrary(llvm::StringRef libStem) const {
    llvm::SmallString<128> Absolute(libStem);
    llvm::sys::fs::make_absolute(Absolute);
    bool isAbsolute = libStem == Absolute;

    // If it is an absolute path, don't try iterate over the paths.
    if (isAbsolute) {
      if (isSharedLib(libStem))
        return normalizePath(libStem);
      else
        return "";
    }

    std::string foundName = lookupLibMaybeAddExt(libStem);
    if (foundName.empty() && !libStem.startswith("lib")) {
      // try with "lib" prefix:
      foundName = lookupLibMaybeAddExt("lib" + libStem.str());
    }

    if (isSharedLib(foundName))
      return normalizePath(foundName);
    return "";
  }

  DynamicLibraryManager::LoadLibResult
  DynamicLibraryManager::loadLibrary(const std::string& libStem,
                                     bool permanent, bool resolved) {
    std::string       lResolved;
    const std::string &canonicalLoadedLib = resolved ? libStem : lResolved;
    if (!resolved) {
      lResolved = lookupLibrary(libStem);
      if (lResolved.empty())
        return kLoadLibNotFound;
    }

    if (m_LoadedLibraries.find(canonicalLoadedLib) != m_LoadedLibraries.end())
      return kLoadLibAlreadyLoaded;

    std::string errMsg;
    // TODO: !permanent case
#if defined(LLVM_ON_WIN32)
    HMODULE dyLibHandle = LoadLibraryEx(canonicalLoadedLib.c_str(), NULL,
                                        DONT_RESOLVE_DLL_REFERENCES);
    if (!dyLibHandle)
      platform::GetLastErrorAsString(errMsg, "LoadLibraryEx");
#else
    const void* dyLibHandle = dlopen(canonicalLoadedLib.c_str(),
                                     RTLD_LAZY|RTLD_GLOBAL);
    if (const char* DyLibError = dlerror()) {
      errMsg = DyLibError;
    }
#endif
    if (!dyLibHandle) {
      llvm::errs() << "cling::DynamicLibraryManager::loadLibrary(): " << errMsg
                   << '\n';
      return kLoadLibLoadError;
    }
    else if (InterpreterCallbacks* C = getCallbacks())
      C->LibraryLoaded(dyLibHandle, canonicalLoadedLib);

    std::pair<DyLibs::iterator, bool> insRes
      = m_DyLibs.insert(std::pair<DyLibHandle, std::string>(dyLibHandle,
                                                            canonicalLoadedLib));
    if (!insRes.second)
      return kLoadLibAlreadyLoaded;
    m_LoadedLibraries.insert(canonicalLoadedLib);
    return kLoadLibSuccess;
  }

  void DynamicLibraryManager::unloadLibrary(llvm::StringRef libStem) {
    std::string canonicalLoadedLib = lookupLibrary(libStem);
    if (!isLibraryLoaded(canonicalLoadedLib))
      return;

    DyLibHandle dyLibHandle = 0;
    for (DyLibs::const_iterator I = m_DyLibs.begin(), E = m_DyLibs.end();
         I != E; ++I) {
      if (I->second == canonicalLoadedLib) {
        dyLibHandle = I->first;
        break;
      }
    }

    std::string errMsg;
    // TODO: !permanent case
#if defined(LLVM_ON_WIN32)
    if (FreeLibrary((HMODULE)dyLibHandle) == 0)
      platform::GetLastErrorAsString(errMsg, "FreeLibrary");
#else
    dlclose(const_cast<void*>(dyLibHandle));
    if (const char* DyLibError = dlerror()) {
      errMsg = DyLibError;
    }
#endif
    if (!errMsg.empty()) {
      llvm::errs() << "cling::DynamicLibraryManager::unloadLibrary(): "
                   << errMsg << '\n';
    }

    if (InterpreterCallbacks* C = getCallbacks())
      C->LibraryUnloaded(dyLibHandle, canonicalLoadedLib);

    m_DyLibs.erase(dyLibHandle);
    m_LoadedLibraries.erase(canonicalLoadedLib);
  }

  bool DynamicLibraryManager::isLibraryLoaded(llvm::StringRef fullPath) const {
    std::string canonPath = normalizePath(fullPath);
    if (m_LoadedLibraries.find(canonPath) != m_LoadedLibraries.end())
      return true;
    return false;
  }

  void DynamicLibraryManager::ExposeHiddenSharedLibrarySymbols(void* handle) {
    llvm::sys::DynamicLibrary::addPermanentLibrary(const_cast<void*>(handle));
  }
} // end namespace cling
