//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/DynamicLibraryManager.h"
#include "cling/Interpreter/InvocationOptions.h"

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <stdlib.h>

#ifdef WIN32
#include <Windows.h>
#else
#include <limits.h> /* PATH_MAX */
#include <dlfcn.h>
#endif

namespace cling {
  DynamicLibraryManager::DynamicLibraryManager(const InvocationOptions& Opts) 
    : m_Opts(Opts) { }

  DynamicLibraryManager::~DynamicLibraryManager() {}

  static bool isSharedLib(llvm::StringRef LibName, bool& exists) {
    using namespace llvm::sys::fs;
    file_magic Magic;
    llvm::error_code Error = identify_magic(LibName, Magic);
    exists = (Error == llvm::errc::success);
    return exists &&
#ifdef __APPLE__
        (Magic == file_magic::macho_fixed_virtual_memory_shared_lib
         || Magic == file_magic::macho_dynamically_linked_shared_lib
         || Magic == file_magic::macho_dynamically_linked_shared_lib_stub)
#elif defined(LLVM_ON_UNIX)
        Magic == file_magic::elf_shared_object
#elif defined(LLVM_ON_WIN32)
# error "Windows DLLs  not yet implemented!"
        //Magic == file_magic::pecoff_executable?
#else
# error "Unsupported platform."
#endif
      ;
  }

  static void
  findSharedLibrary(llvm::StringRef fileStem,
                    const llvm::SmallVectorImpl<std::string>& Paths,
                    llvm::SmallString<512>& FoundDyLib,
                    bool& exists, bool& isDyLib) {
    for (llvm::SmallVectorImpl<std::string>::const_iterator
        IPath = Paths.begin(), EPath = Paths.end(); IPath != EPath; ++IPath) {
      llvm::SmallString<512> ThisPath(*IPath);
      llvm::sys::path::append(ThisPath, fileStem);
      isDyLib = isSharedLib(ThisPath.str(), exists);
      if (isDyLib)
        ThisPath.swap(FoundDyLib);
      if (exists)
        return;
    }
  }

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

    Paths.push_back("/usr/local/lib/");
    Paths.push_back("/usr/X11R6/lib/");
    Paths.push_back("/usr/lib/");
    Paths.push_back("/lib/");
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

  DynamicLibraryManager::LoadLibResult
  DynamicLibraryManager::tryLinker(const std::string& filename, bool permanent,
                                   bool isAbsolute, bool& exists,
                                   bool& isDyLib) {
    using namespace llvm::sys;
    exists = false;
    isDyLib = false;

    llvm::SmallString<512> FoundDyLib;

    if (isAbsolute) {
      isDyLib = isSharedLib(filename, exists);
      if (isDyLib)
        FoundDyLib = filename;
    } else {
      llvm::SmallVector<std::string, 16>
        SearchPaths(m_Opts.LibSearchPath.begin(), m_Opts.LibSearchPath.end());
      GetSystemLibraryPaths(SearchPaths);

      findSharedLibrary(filename, SearchPaths, FoundDyLib, exists, isDyLib);

      if (!exists) {
        // Add DyLib extension:
        llvm::SmallString<512> filenameWithExt(filename);
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
        findSharedLibrary(filenameWithExt, SearchPaths, FoundDyLib, exists,
                          isDyLib);
#ifdef __APPLE__
        if (!exists) {
           filenameWithExt.erase(IStemEnd + 1, filenameWithExt.end());
           filenameWithExt += ".dylib";
           findSharedLibrary(filenameWithExt, SearchPaths, FoundDyLib, exists,
                             isDyLib);
        }
#endif
      }
    }

    if (!isDyLib)
      return kLoadLibError;
    
    assert(!FoundDyLib.empty() && "The shared lib exists but can't find it!");

    // get canonical path name and check if already loaded
#ifdef WIN32
    llvm::SmallString<_MAX_PATH> FullPath("");
    char *res = _fullpath((char *)FullPath.data(), FoundDyLib.c_str(), _MAX_PATH);
#else
    llvm::SmallString<PATH_MAX+1> FullPath("");
    char *res = realpath(FoundDyLib.c_str(), (char *)FullPath.data());
#endif
    if (res == 0) {
      llvm::errs() << "cling::Interpreter::tryLinker(): error getting real (canonical) path\n";
      return kLoadLibError;
    }
    FullPath.set_size(strlen(res));
    if (m_loadedLibraries.find(FullPath) != m_loadedLibraries.end())
      return kLoadLibExists;

    // TODO: !permanent case
#ifdef WIN32
# error "Windows DLL opening still needs to be implemented!"
    void* dyLibHandle = needs to be implemented!;
    std::string errMsg;
#else
    const void* dyLibHandle = dlopen(FullPath.c_str(), RTLD_LAZY|RTLD_GLOBAL);
    std::string errMsg;
    if (const char* DyLibError = dlerror()) {
      errMsg = DyLibError;
    }
#endif
    if (!dyLibHandle) {
      llvm::errs() << "cling::Interpreter::tryLinker(): " << errMsg << '\n';
      return kLoadLibError;
    }
    std::pair<DyLibs::iterator, bool> insRes
      = m_DyLibs.insert(std::pair<DyLibHandle, std::string>(dyLibHandle,
                                                            FullPath.str()));
    if (!insRes.second)
      return kLoadLibExists;
    m_loadedLibraries.insert(FullPath);
    return kLoadLibSuccess;
  }

  DynamicLibraryManager::LoadLibResult
  DynamicLibraryManager::loadLibrary(const std::string& filename,
                                     bool permanent, bool* tryCode) {
    llvm::SmallString<128> Absolute((llvm::StringRef(filename)));
    llvm::sys::fs::make_absolute(Absolute);
    bool isAbsolute = filename == Absolute.c_str();
    bool exists = false;
    bool isDyLib = false;
    LoadLibResult res = tryLinker(filename, permanent, isAbsolute, exists,
                                  isDyLib);
    if (tryCode) {
      *tryCode = !isDyLib;
      if (isAbsolute)
        *tryCode &= exists;
    }
    if (exists)
      return res;

    if (!isAbsolute && filename.compare(0, 3, "lib")) {
      // try with "lib" prefix:
      res = tryLinker("lib" + filename, permanent, false, exists, isDyLib);
      if (tryCode) {
        *tryCode = !isDyLib;
        if (isAbsolute)
          *tryCode &= exists;
      }
      if (res != kLoadLibError)
        return res;
    }
    return kLoadLibError;
  }

  bool
  DynamicLibraryManager::isDynamicLibraryLoaded(llvm::StringRef fullPath) const {
    // get canonical path name and check if already loaded
#ifdef WIN32
    char buf[_MAX_PATH];
    char *res = _fullpath(buf, fullPath.str().c_str(), _MAX_PATH);
#else
    char buf[PATH_MAX+1];
    char *res = realpath(fullPath.str().c_str(), buf);
#endif
    if (res == 0) {
      llvm::errs() << "cling::Interpreter::isDynamicLibraryLoaded(): error getting real (canonical) path\n";
      return false;
    }
    if (m_loadedLibraries.find(buf) != m_loadedLibraries.end()) return true;
    return false;
  }

  void DynamicLibraryManager::ExposeHiddenSharedLibrarySymbols(void* handle) {
    llvm::sys::DynamicLibrary::addPermanentLibrary(const_cast<void*>(handle));
  }
} // end namespace cling
