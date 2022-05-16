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
#include "cling/Utils/Paths.h"
#include "cling/Utils/Platform.h"
#include "cling/Utils/Output.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Path.h"

#include <system_error>
#include <sys/stat.h>

// FIXME: Implement debugging output stream in cling.
constexpr unsigned DEBUG = 0;

namespace cling {
  DynamicLibraryManager::DynamicLibraryManager()  {
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
  #elif defined(_WIN32)
      "PATH",
  #endif
    };

    // Behaviour is to not add paths that don't exist...In an interpreted env
    // does this make sense? Path could pop into existance at any time.
    for (const char* Var : kSysLibraryEnv) {
      if (const char* Env = ::getenv(Var)) {
        llvm::SmallVector<llvm::StringRef, 10> CurPaths;
        SplitPaths(Env, CurPaths, utils::kPruneNonExistant, platform::kEnvDelim);
        for (const auto& Path : CurPaths)
          addSearchPath(Path);
      }
    }

    // $CWD is the last user path searched.
    addSearchPath(".");

    llvm::SmallVector<std::string, 64> SysPaths;
    platform::GetSystemLibraryPaths(SysPaths);

    for (const std::string& P : SysPaths)
      addSearchPath(P, /*IsUser*/ false);
  }

  ///\returns substitution of pattern in the front of original with replacement
  /// Example: substFront("@rpath/abc", "@rpath/", "/tmp") -> "/tmp/abc"
  static std::string substFront(llvm::StringRef original, llvm::StringRef pattern,
                                llvm::StringRef replacement) {
    if (!original.startswith_lower(pattern))
      return original.str();
    llvm::SmallString<512> result(replacement);
    result.append(original.drop_front(pattern.size()));
    return result.str();
  }

  ///\returns substitution of all known linker variables in \c original
  static std::string substAll(llvm::StringRef original,
                              llvm::StringRef libLoader) {

    // Handle substitutions (MacOS):
    // @rpath - This function does not substitute @rpath, becouse
    //          this variable is already handled by lookupLibrary where
    //          @rpath is replaced with all paths from RPATH one by one.
    // @executable_path - Main program path.
    // @loader_path - Loader library (or main program) path.
    //
    // Handle substitutions (Linux):
    // https://man7.org/linux/man-pages/man8/ld.so.8.html
    // $origin - Loader library (or main program) path.
    // $lib - lib lib64
    // $platform - x86_64 AT_PLATFORM

    std::string result;
#ifdef __APPLE__
    llvm::SmallString<512> mainExecutablePath(llvm::sys::fs::getMainExecutable(nullptr, nullptr));
    llvm::sys::path::remove_filename(mainExecutablePath);
    llvm::SmallString<512> loaderPath;
    if (libLoader.empty()) {
      loaderPath = mainExecutablePath;
    } else {
      loaderPath = libLoader.str();
      llvm::sys::path::remove_filename(loaderPath);
    }

    result = substFront(original, "@executable_path", mainExecutablePath);
    result = substFront(result, "@loader_path", loaderPath);
    return result;
#else
    llvm::SmallString<512> loaderPath;
    if (libLoader.empty()) {
      loaderPath = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
    } else {
      loaderPath = libLoader.str();
    }
    llvm::sys::path::remove_filename(loaderPath);

    result = substFront(original, "$origin", loaderPath);
    //result = substFront(result, "$lib", true?"lib":"lib64");
    //result = substFront(result, "$platform", "x86_64");
    return result;
#endif
  }

  std::string
  DynamicLibraryManager::lookupLibInPaths(llvm::StringRef libStem,
                                          llvm::SmallVector<llvm::StringRef,2> RPath /*={}*/,
                                          llvm::SmallVector<llvm::StringRef,2> RunPath /*={}*/,
                                          llvm::StringRef libLoader /*=""*/) const {

    if (DEBUG > 7) {
      cling::errs() << "Dyld::lookupLibInPaths:" << libStem.str() <<
        ", ..., libLodaer=" << libLoader << "\n";
    }

    // Lookup priority is: RPATH, LD_LIBRARY_PATH/m_SearchPaths, RUNPATH
    // See: https://en.wikipedia.org/wiki/Rpath
    // See: https://amir.rachum.com/blog/2016/09/17/shared-libraries/

    if (DEBUG > 7) {
      cling::errs() << "Dyld::lookupLibInPaths: \n";
      cling::errs() << ":: RPATH\n";
      for (auto Info : RPath) {
        cling::errs() << ":::: " << Info.str() << "\n";
      }
      cling::errs() << ":: SearchPaths (LD_LIBRARY_PATH, etc...)\n";
      for (auto Info : getSearchPaths()) {
        cling::errs() << ":::: " << Info.Path << ", user=" << (Info.IsUser?"true":"false") << "\n";
      }
      cling::errs() << ":: RUNPATH\n";
      for (auto Info : RunPath) {
        cling::errs() << ":::: " << Info.str() << "\n";
      }
    }

    llvm::SmallString<512> ThisPath;
    // RPATH
    for (auto Info : RPath) {
      ThisPath = substAll(Info, libLoader);
      llvm::sys::path::append(ThisPath, libStem);
      // to absolute path?
      if (DEBUG > 7) {
        cling::errs() << "## Try: " << ThisPath;
      }
      if (isSharedLibrary(ThisPath.str())) {
        if (DEBUG > 7) {
          cling::errs() << " ... Found (in RPATH)!\n";
        }
        return ThisPath.str();
      }
    }
    // m_SearchPaths
    for (const SearchPathInfo& Info : m_SearchPaths) {
      ThisPath = Info.Path;
      llvm::sys::path::append(ThisPath, libStem);
      // to absolute path?
      if (DEBUG > 7) {
        cling::errs() << "## Try: " << ThisPath;
      }
      if (isSharedLibrary(ThisPath.str())) {
        if (DEBUG > 7) {
          cling::errs() << " ... Found (in SearchPaths)!\n";
        }
        return ThisPath.str();
      }
    }
    // RUNPATH
    for (auto Info : RunPath) {
      ThisPath = substAll(Info, libLoader);
      llvm::sys::path::append(ThisPath, libStem);
      // to absolute path?
      if (DEBUG > 7) {
        cling::errs() << "## Try: " << ThisPath;
      }
      if (isSharedLibrary(ThisPath.str())) {
        if (DEBUG > 7) {
          cling::errs() << " ... Found (in RUNPATH)!\n";
        }
        return ThisPath.str();
      }
    }

    if (DEBUG > 7) {
      cling::errs() << "## NotFound!!!\n";
    }

    return "";
  }

  std::string
  DynamicLibraryManager::lookupLibMaybeAddExt(llvm::StringRef libStem,
                                              llvm::SmallVector<llvm::StringRef,2> RPath /*={}*/,
                                              llvm::SmallVector<llvm::StringRef,2> RunPath /*={}*/,
                                              llvm::StringRef libLoader /*=""*/) const {

    using namespace llvm::sys;

    if (DEBUG > 7) {
      cling::errs() << "Dyld::lookupLibMaybeAddExt: " << libStem.str() <<
        ", ..., libLoader=" << libLoader << "\n";
    }

    std::string foundDyLib = lookupLibInPaths(libStem, RPath, RunPath, libLoader);

    if (foundDyLib.empty()) {
      // Add DyLib extension:
      llvm::SmallString<512> filenameWithExt(libStem);
#if defined(LLVM_ON_UNIX)
#ifdef __APPLE__
      llvm::SmallString<512>::iterator IStemEnd = filenameWithExt.end() - 1;
#endif
      static const char* DyLibExt = ".so";
#elif defined(_WIN32)
      static const char* DyLibExt = ".dll";
#else
# error "Unsupported platform."
#endif
      filenameWithExt += DyLibExt;
      foundDyLib = lookupLibInPaths(filenameWithExt, RPath, RunPath, libLoader);
#ifdef __APPLE__
      if (foundDyLib.empty()) {
        filenameWithExt.erase(IStemEnd + 1, filenameWithExt.end());
        filenameWithExt += ".dylib";
        foundDyLib = lookupLibInPaths(filenameWithExt, RPath, RunPath, libLoader);
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

  std::string RPathToStr2(llvm::SmallVector<llvm::StringRef,2> V) {
    std::string result;
    for (auto item : V)
      result += item.str() + ",";
    if (!result.empty())
      result.pop_back();
    return result;
  }

  std::string
  DynamicLibraryManager::lookupLibrary(llvm::StringRef libStem,
                                       llvm::SmallVector<llvm::StringRef,2> RPath /*={}*/,
                                       llvm::SmallVector<llvm::StringRef,2> RunPath /*={}*/,
//                                       llvm::StringRef RPath /*=""*/,
//                                       llvm::StringRef RunPath /*=""*/,
                                       llvm::StringRef libLoader /*=""*/,
                                       bool variateLibStem /*=true*/) const {
    if (DEBUG > 7) {
      cling::errs() << "Dyld::lookupLibrary: " << libStem.str() << ", " <<
        RPathToStr2(RPath) << ", " << RPathToStr2(RunPath) << ", " << libLoader.str() << "\n";
    }

    // If it is an absolute path, don't try iterate over the paths.
    if (llvm::sys::path::is_absolute(libStem)) {
      if (isSharedLibrary(libStem))
        return normalizePath(libStem);
      else
        return std::string();
    }

    // Subst all known linker variables ($origin, @rpath, etc.)
#ifdef __APPLE__
    // On MacOS @rpath is preplaced by all paths in RPATH one by one.
    if (libStem.startswith_lower("@rpath")) {
      for (auto& P : RPath) {
        std::string result = substFront(libStem, "@rpath", P);
        if (isSharedLibrary(result))
          return normalizePath(result);
      }
    } else {
#endif
      std::string result = substAll(libStem, libLoader);
      if (isSharedLibrary(result))
        return normalizePath(result);
#ifdef __APPLE__
    }
#endif

    // Expand libStem with paths, extensions, etc.
    std::string foundName;
    if (variateLibStem) {
      foundName = lookupLibMaybeAddExt(libStem, RPath, RunPath, libLoader);
      if (foundName.empty()) {
        llvm::StringRef libStemName = llvm::sys::path::filename(libStem);
        if (!libStemName.startswith("lib")) {
          // try with "lib" prefix:
          foundName = lookupLibMaybeAddExt(
             libStem.str().insert(libStem.size()-libStemName.size(), "lib"),
             RPath,
             RunPath,
             libLoader
          );
        }
      }
    } else {
      foundName = lookupLibInPaths(libStem, RPath, RunPath, libLoader);
    }

    if (!foundName.empty())
      return platform::NormalizePath(foundName);

    return std::string();
  }

  DynamicLibraryManager::LoadLibResult
  DynamicLibraryManager::loadLibrary(llvm::StringRef libStem,
                                     bool permanent, bool resolved) {
    if (DEBUG > 7) {
      cling::errs() << "Dyld::loadLibrary: " << libStem.str() << ", " <<
        (permanent ? "permanent" : "not-permanent") << ", " <<
        (resolved ? "resolved" : "not-resolved") << "\n";
    }

    std::string canonicalLoadedLib;
    if (resolved) {
      canonicalLoadedLib = libStem.str();
    } else {
      canonicalLoadedLib = lookupLibrary(libStem);
      if (canonicalLoadedLib.empty())
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
        if (C->LibraryLoadingFailed(errMsg, libStem.str(), permanent, resolved))
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

  void DynamicLibraryManager::dump(llvm::raw_ostream* S /*= nullptr*/) const {
    llvm::raw_ostream &OS = S ? *S : cling::outs();

    // FIXME: print in a stable order the contents of m_SearchPaths
    for (const auto& Info : getSearchPaths()) {
      if (!Info.IsUser)
        OS << "[system] ";
      OS << Info.Path.c_str() << "\n";
    }
  }

  void DynamicLibraryManager::ExposeHiddenSharedLibrarySymbols(void* handle) {
    llvm::sys::DynamicLibrary::addPermanentLibrary(const_cast<void*>(handle));
  }

  bool DynamicLibraryManager::isSharedLibrary(llvm::StringRef libFullPath,
                                              bool* exists /*=0*/) {
    using namespace llvm;

    auto filetype = sys::fs::get_file_type(libFullPath, /*Follow*/ true);
    if (filetype != sys::fs::file_type::regular_file) {
      if (exists) {
        // get_file_type returns status_error also in case of file_not_found.
        *exists = filetype != sys::fs::file_type::status_error;
      }
      return false;
    }

    file_magic Magic;
    const std::error_code Error = identify_magic(libFullPath, Magic);
    if (exists)
      *exists = !Error;

    bool result = !Error &&
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
#elif defined(_WIN32)
      // We should only include dll libraries without including executables,
      // object code and others...
      (Magic == file_magic::pecoff_executable &&
       platform::IsDLL(libFullPath.str()))
#else
# error "Unsupported platform."
#endif
      ;

      return result;
  }

} // end namespace cling
