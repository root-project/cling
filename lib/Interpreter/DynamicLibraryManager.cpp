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

#ifdef WIN32
#include <Windows.h>
#else
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
#ifdef LTDL_SHLIBPATH_VAR
    char* env_var = getenv(LTDL_SHLIBPATH_VAR);
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

#endif
    // FIXME: Should this look at LD_LIBRARY_PATH too?
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
        static const char* DyLibExt = ".so";
#elif defined(LLVM_ON_WIN32)
        static const char* DyLibExt = ".dll";
#else
# error "Unsupported platform."
#endif
        filenameWithExt += DyLibExt;
        findSharedLibrary(filenameWithExt, SearchPaths, FoundDyLib, exists,
                          isDyLib);
      }
    }

    if (!isDyLib)
      return kLoadLibError;
    
    assert(!FoundDyLib.empty() && "The shared lib exists but can't find it!");

    // TODO: !permanent case
#ifdef WIN32
# error "Windows DLL opening still needs to be implemented!"
    void* dyLibHandle = needs to be implemented!;
    std::string errMsg;
#else
    const void* dyLibHandle
      = dlopen(FoundDyLib.c_str(), RTLD_LAZY|RTLD_GLOBAL);
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
                                                            FoundDyLib.str()));
    if (!insRes.second)
      return kLoadLibExists;
    return kLoadLibSuccess;
  }

  DynamicLibraryManager::LoadLibResult
  DynamicLibraryManager::loadLibrary(const std::string& filename,
                                     bool permanent, bool* tryCode) {
    // If it's not an absolute path, prepend "lib"
    llvm::SmallVector<char, 128> Absolute(filename.c_str(),
                                          filename.c_str() + filename.length());
    Absolute.push_back(0);
    llvm::sys::fs::make_absolute(Absolute);
    bool isAbsolute = filename == Absolute.data();
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
  DynamicLibraryManager::isDynamicLibraryLoaded(llvm::StringRef fullPath) const{
    for(DyLibs::const_iterator I = m_DyLibs.begin(), E = m_DyLibs.end(); 
        I != E; ++I) {
      if (fullPath.equals((I->second)))
        return true;
    }
    return false;
  }


  void DynamicLibraryManager::ExposeHiddenSharedLibrarySymbols(void* handle) {
    llvm::sys::DynamicLibrary::addPermanentLibrary(const_cast<void*>(handle));
  }
} // end namespace cling
