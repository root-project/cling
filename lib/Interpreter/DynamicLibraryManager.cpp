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
#include "cling/Utils/Output.h"

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Path.h"

#include <system_error>
#include <sys/stat.h>

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

    // Behaviour is to not add paths that don't exist...In an interpreted env
    // does this make sense? Path could pop into existance at any time.
    for (const char* Var : kSysLibraryEnv) {
      if (Opts.Verbose())
        cling::log() << "Adding library paths from '" << Var << "':\n";
      if (const char* Env = ::getenv(Var)) {
        llvm::SmallVector<llvm::StringRef, 10> CurPaths;
        SplitPaths(Env, CurPaths, utils::kPruneNonExistant, platform::kEnvDelim,
                   Opts.Verbose());
        for (const auto& Path : CurPaths)
          m_SystemSearchPaths.push_back(Path.str());
      }
    }

    platform::GetSystemLibraryPaths(m_SystemSearchPaths);

    // This will currently be the last path searched, should it be pushed to
    // the front of the line, or even to the front of user paths?
    m_SystemSearchPaths.push_back(".");
  }

  DynamicLibraryManager::~DynamicLibraryManager() {}

  static bool isSharedLib(llvm::StringRef LibName, bool* exists = 0) {
    using namespace llvm;
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
      return std::string();

    // get canonical path name and check if already loaded
    const std::string Path = platform::NormalizePath(foundDyLib);
    if (Path.empty()) {
      cling::errs() << "cling::DynamicLibraryManager::lookupLibMaybeAddExt(): "
        "error getting real (canonical) path of library " << foundDyLib << '\n';
      return foundDyLib;
    }
    return Path;
  }

  std::string DynamicLibraryManager::normalizePath(llvm::StringRef path) {
    // Make the path canonical if the file exists.
    const std::string Path = path.str();
    struct stat buffer;
    if (::stat(Path.c_str(), &buffer) != 0)
      return std::string();

    const std::string NPath = platform::NormalizePath(Path);
    if (NPath.empty())
      cling::log() << "Could not normalize: '" << Path << "'";
    return NPath;
  }

  std::string
  DynamicLibraryManager::lookupLibrary(llvm::StringRef libStem) const {
    // If it is an absolute path, don't try iterate over the paths.
    if (llvm::sys::path::is_absolute(libStem)) {
      if (isSharedLib(libStem))
        return normalizePath(libStem);
      else
        return std::string();
    }

    std::string foundName = lookupLibMaybeAddExt(libStem);
    if (foundName.empty() && !libStem.startswith("lib")) {
      // try with "lib" prefix:
      foundName = lookupLibMaybeAddExt("lib" + libStem.str());
    }

    if (!foundName.empty() && isSharedLib(foundName))
      return platform::NormalizePath(foundName);

    return std::string();
  }

  DynamicLibraryManager::LoadLibResult
  DynamicLibraryManager::loadLibrary(const std::string& libStem,
                                     bool permanent, bool resolved) {
    std::string lResolved;
    const std::string& canonicalLoadedLib = resolved ? libStem : lResolved;
    if (!resolved) {
      lResolved = lookupLibrary(libStem);
      if (lResolved.empty())
        return kLoadLibNotFound;
    }

    if (m_LoadedLibraries.find(canonicalLoadedLib) != m_LoadedLibraries.end())
      return kLoadLibAlreadyLoaded;

    // TODO: !permanent case

    std::string errMsg;
    DyLibHandle dyLibHandle = platform::DLOpen(canonicalLoadedLib, &errMsg);
    if (!dyLibHandle) {
      // We emit callback to LibraryLoadingFailed when we get error with error message.
      if (InterpreterCallbacks* C = getCallbacks()) {
        if (C->LibraryLoadingFailed(errMsg, libStem, permanent, resolved))
          return kLoadLibSuccess;
      }

      cling::errs() << "cling::DynamicLibraryManager::loadLibrary(): " << errMsg
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

    // TODO: !permanent case

    std::string errMsg;
    platform::DLClose(dyLibHandle, &errMsg);
    if (!errMsg.empty()) {
      cling::errs() << "cling::DynamicLibraryManager::unloadLibrary(): "
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
