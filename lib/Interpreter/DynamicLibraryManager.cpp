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

  static llvm::sys::Path
  findSharedLibrary(llvm::StringRef fileStem,
                    const llvm::SmallVectorImpl<llvm::sys::Path>& Paths,
                    bool& exists, bool& isDyLib) {
    for (llvm::SmallVectorImpl<llvm::sys::Path>::const_iterator
           IPath = Paths.begin(), EPath = Paths.end(); IPath != EPath; ++IPath){
      llvm::sys::Path ThisPath(*IPath);
      ThisPath.appendComponent(fileStem);
      exists = llvm::sys::fs::exists(ThisPath.str());
      if (exists && ThisPath.isDynamicLibrary()) {
        isDyLib = true;
        return ThisPath;
      }
    }
    return llvm::sys::Path();
  }

  DynamicLibraryManager::LoadLibResult
  DynamicLibraryManager::tryLinker(const std::string& filename, bool permanent,
                                   bool isAbsolute, bool& exists, 
                                   bool& isDyLib) {
    using namespace llvm::sys;
    exists = false;
    isDyLib = false;

    llvm::sys::Path FoundDyLib;

    if (isAbsolute) {
      exists = llvm::sys::fs::exists(filename.c_str());
      if (exists && Path(filename).isDynamicLibrary()) {
        isDyLib = true;
        FoundDyLib = filename;
      }
    } else {
      llvm::SmallVector<Path, 16>
      SearchPaths(m_Opts.LibSearchPath.begin(), m_Opts.LibSearchPath.end());

      std::vector<Path> SysSearchPaths;
      Path::GetSystemLibraryPaths(SysSearchPaths);
      SearchPaths.append(SysSearchPaths.begin(), SysSearchPaths.end());

      FoundDyLib = findSharedLibrary(filename, SearchPaths, exists, isDyLib);

      std::string filenameWithExt(filename);
      filenameWithExt += ("." + Path::GetDLLSuffix()).str();
      if (!exists) {
        // Add DyLib extension:
        FoundDyLib = findSharedLibrary(filenameWithExt, SearchPaths, exists,
                                       isDyLib);
      }
    }

    if (!isDyLib)
      return kLoadLibError;
    
    assert(!FoundDyLib.isEmpty() && "The shared lib exists but can't find it!");

    // TODO: !permanent case
#ifdef WIN32
    void* dyLibHandle = needs to be implemented!;
    std::string errMsg;
#else
    const void* dyLibHandle
      = dlopen(FoundDyLib.str().c_str(), RTLD_LAZY|RTLD_GLOBAL);
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
