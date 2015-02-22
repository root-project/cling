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
#include <shlobj.h>
#else
#include <limits.h> /* PATH_MAX */
#include <dlfcn.h>
#endif

namespace {
#if defined(LLVM_ON_UNIX)
  static void GetSystemLibraryPaths(llvm::SmallVectorImpl<std::string>& Paths) {
    char* env_var = getenv("LD_LIBRARY_PATH");
#if __APPLE__
    if (!env_var)
      env_var = getenv("DYLD_LIBRARY_PATH");
    if (!env_var)
      env_var = getenv("DYLD_FALLBACK_LIBRARY_PATH");
#endif
    if (env_var != 0) {
      static const char PathSeparator = ':';
      const char* at = env_var;
      const char* delim = strchr(at, PathSeparator);
      while (delim != 0) {
        std::string tmp(at, size_t(delim-at));
        if (llvm::sys::fs::is_directory(tmp.c_str()))
          Paths.push_back(tmp);
        at = delim + 1;
        delim = strchr(at, PathSeparator);
      }

      if (*at != 0)
        if (llvm::sys::fs::is_directory(llvm::StringRef(at)))
          Paths.push_back(at);
    }
#if defined(__APPLE__) || defined(__CYGWIN__)
    Paths.push_back("/usr/local/lib/");
    Paths.push_back("/usr/X11R6/lib/");
    Paths.push_back("/usr/lib/");
    Paths.push_back("/lib/");

    Paths.push_back("/lib/x86_64-linux-gnu/");
    Paths.push_back("/usr/local/lib64/");
    Paths.push_back("/usr/lib64/");
    Paths.push_back("/lib64/");
#else
    static bool initialized = false;
    static std::vector<std::string> SysPaths;
    if (!initialized) {
      // trick to get the system search path
      std::string cmd("LD_DEBUG=libs LD_PRELOAD=DOESNOTEXIST ls 2>&1");
      FILE *pf = popen(cmd.c_str (), "r");
      std::string result = "";
      std::string sys_path = "";
      char buffer[128];
      while (!feof(pf)) {
        if (fgets(buffer, 128, pf) != NULL)
          result += buffer;
      }
      pclose(pf);
      std::size_t from
        = result.find("search path=", result.find("(LD_LIBRARY_PATH)"));
      std::size_t to = result.find("(system search path)");
      if (from != std::string::npos && to != std::string::npos) {
        from += 12;
        sys_path = result.substr(from, to-from);
        sys_path.erase(std::remove_if(sys_path.begin(), sys_path.end(), isspace),
                       sys_path.end());
        sys_path += ':';
      }
      static const char PathSeparator = ':';
      const char* at = sys_path.c_str();
      const char* delim = strchr(at, PathSeparator);
      while (delim != 0) {
        std::string tmp(at, size_t(delim-at));
        if (llvm::sys::fs::is_directory(tmp.c_str()))
          SysPaths.push_back(tmp);
        at = delim + 1;
        delim = strchr(at, PathSeparator);
      }
      initialized = true;
    }

    for (std::vector<std::string>::const_iterator I = SysPaths.begin(),
           E = SysPaths.end(); I != E; ++I)
      Paths.push_back((*I).c_str());
#endif
  }
#elif defined(LLVM_ON_WIN32)
  static void GetSystemLibraryPaths(llvm::SmallVectorImpl<std::string>& Paths) {
    char buff[MAX_PATH];
    // Generic form of C:\Windows\System32
    HRESULT res =  SHGetFolderPathA(NULL,
                                    CSIDL_FLAG_CREATE | CSIDL_SYSTEM,
                                    NULL,
                                    SHGFP_TYPE_CURRENT,
                                    buff);
    if (res != S_OK) {
      assert(0 && "Failed to get system directory");
      return;
    }
    Paths.push_back(buff);

    // Reset buff.
    buff[0] = 0;
    // Generic form of C:\Windows
    res =  SHGetFolderPathA(NULL,
                            CSIDL_FLAG_CREATE | CSIDL_WINDOWS,
                            NULL,
                            SHGFP_TYPE_CURRENT,
                            buff);
    if (res != S_OK) {
      assert(0 && "Failed to get windows directory");
      return;
    }
    Paths.push_back(buff);
  }
#else
# error "Unsupported platform."
#endif

}

namespace cling {
  DynamicLibraryManager::DynamicLibraryManager(const InvocationOptions& Opts)
    : m_Opts(Opts), m_Callbacks(0) {
    GetSystemLibraryPaths(m_SystemSearchPaths);
    m_SystemSearchPaths.push_back(".");
  }

  DynamicLibraryManager::~DynamicLibraryManager() {}

  static bool isSharedLib(llvm::StringRef LibName, bool* exists = 0) {
    using namespace llvm::sys::fs;
    file_magic Magic;
    std::error_code Error = identify_magic(LibName, Magic);
    bool onDisk = (Error != std::errc::no_such_file_or_directory);
    if (exists)
      *exists = onDisk;

    return onDisk &&
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
      (Magic == file_magic::pecoff_executable)
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
                                     bool permanent) {
    std::string canonicalLoadedLib = lookupLibrary(libStem);
    if (canonicalLoadedLib.empty())
      return kLoadLibNotFound;

    if (m_LoadedLibraries.find(canonicalLoadedLib) != m_LoadedLibraries.end())
      return kLoadLibAlreadyLoaded;

    std::string errMsg;
    // TODO: !permanent case
#if defined(LLVM_ON_WIN32)
    HMODULE dyLibHandle = LoadLibraryEx(canonicalLoadedLib.c_str(), NULL,
                                        DONT_RESOLVE_DLL_REFERENCES);
    errMsg = "LoadLibraryEx: GetLastError() returned ";
    errMsg += GetLastError();
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
      if (I->second == canonicalLoadedLib)
        dyLibHandle = I->first;
    }

    std::string errMsg;
    // TODO: !permanent case
#if defined(LLVM_ON_WIN32)
    FreeLibrary((HMODULE)dyLibHandle);
    errMsg = "UnoadLibraryEx: GetLastError() returned ";
    errMsg += GetLastError();
#else
    dlclose(const_cast<void*>(dyLibHandle));
    if (const char* DyLibError = dlerror()) {
      errMsg = DyLibError;
    }
#endif
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
