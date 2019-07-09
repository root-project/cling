//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_DYNAMIC_LIBRARY_MANAGER_H
#define CLING_DYNAMIC_LIBRARY_MANAGER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include "llvm/Support/Path.h"

namespace cling {
  class InterpreterCallbacks;
  class InvocationOptions;

  ///\brief A helper class managing dynamic shared objects.
  ///
  class DynamicLibraryManager {
  public:
    ///\brief Describes the result of loading a library.
    ///
    enum LoadLibResult {
      kLoadLibSuccess, ///< library loaded successfully
      kLoadLibAlreadyLoaded,  ///< library was already loaded
      kLoadLibNotFound, ///< library was not found
      kLoadLibLoadError, ///< loading the library failed
      kLoadLibNumResults
    };

    /// Describes the library search paths.
    struct SearchPathInfo {
      /// The search path.
      ///
      std::string Path;

      /// True if the Path is on the LD_LIBRARY_PATH.
      ///
      bool IsUser;
    };
  private:
    typedef const void* DyLibHandle;
    typedef llvm::DenseMap<DyLibHandle, std::string> DyLibs;
    ///\brief DynamicLibraries loaded by this Interpreter.
    ///
    DyLibs m_DyLibs;
    llvm::StringSet<> m_LoadedLibraries;

    ///\brief Contains the list of the current include paths.
    ///
    const InvocationOptions& m_Opts;

    ///\brief System's include path, get initialized at construction time.
    ///
    llvm::SmallVector<SearchPathInfo, 32> m_SearchPaths;

    InterpreterCallbacks* m_Callbacks;

    ///\brief Concatenates current include paths and the system include paths
    /// and performs a lookup for the filename.
    ///\param[in] libStem - The filename being looked up
    ///
    ///\returns the canonical path to the file or empty string if not found
    ///
    std::string lookupLibInPaths(llvm::StringRef libStem) const;


    ///\brief Concatenates current include paths and the system include paths
    /// and performs a lookup for the filename. If still not found it tries to
    /// add the platform-specific extensions (such as so, dll, dylib) and
    /// retries the lookup (from lookupLibInPaths)
    ///\param[in] filename - The filename being looked up
    ///
    ///\returns the canonical path to the file or empty string if not found
    ///
    std::string lookupLibMaybeAddExt(llvm::StringRef filename) const;
  public:
    DynamicLibraryManager(const InvocationOptions& Opts);
    ~DynamicLibraryManager();
    InterpreterCallbacks* getCallbacks() { return m_Callbacks; }
    const InterpreterCallbacks* getCallbacks() const { return m_Callbacks; }
    void setCallbacks(InterpreterCallbacks* C) { m_Callbacks = C; }

    ///\brief Returns the system include paths.
    ///
    ///\returns System include paths.
    ///
    const llvm::SmallVectorImpl<SearchPathInfo>& getSearchPath() {
       return m_SearchPaths;
    }

    ///\brief Looks up a library taking into account the current include paths
    /// and the system include paths.
    ///\param[in] libStem - The filename being looked up
    ///
    ///\returns the canonical path to the file or empty string if not found
    ///
    std::string lookupLibrary(llvm::StringRef libStem) const;

    ///\brief Loads a shared library.
    ///
    ///\param [in] libStem - The file to load.
    ///\param [in] permanent - If false, the file can be unloaded later.
    ///\param [in] resolved - Whether libStem is an absolute path or resolved
    ///               from a previous call to DynamicLibraryManager::lookupLibrary
    ///
    ///\returns kLoadLibSuccess on success, kLoadLibAlreadyLoaded if the library
    /// was already loaded, kLoadLibError if the library cannot be found or any
    /// other error was encountered.
    ///
    LoadLibResult loadLibrary(const std::string& libStem, bool permanent,
                              bool resolved = false);

    void unloadLibrary(llvm::StringRef libStem);

    ///\brief Returns true if the file was a dynamic library and it was already
    /// loaded.
    ///
    bool isLibraryLoaded(llvm::StringRef fullPath) const;

    ///\brief Explicitly tell the execution engine to use symbols from
    ///       a shared library that would otherwise not be used for symbol
    ///       resolution, e.g. because it was dlopened with RTLD_LOCAL.
    ///\param [in] handle - the system specific shared library handle.
    ///
    static void ExposeHiddenSharedLibrarySymbols(void* handle);

    static std::string normalizePath(llvm::StringRef path);

    /// Returns true if file is a shared library.
    ///
    ///\param[in] libFullPath - the full path to file.
    ///
    ///\param[out] exists - sets if the file exists. Useful to distinguish if it
    ///            is a library but of incompatible file format.
    ///
    static bool isSharedLibrary(llvm::StringRef libFullPath, bool* exists = 0);
  };
} // end namespace cling
#endif // CLING_DYNAMIC_LIBRARY_MANAGER_H
