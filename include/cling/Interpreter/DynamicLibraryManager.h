//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: 8a2c4a806b2df22b9d375b228b1ccb502010a74f $
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_DYNAMIC_LIBRARY_MANAGER_H
#define CLING_DYNAMIC_LIBRARY_MANAGER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Support/Path.h"


namespace cling {
  class InvocationOptions;

  ///\brief A helper class managing dynamic shared objects.
  ///
  class DynamicLibraryManager {
  private:
    typedef const void* DyLibHandle;
    typedef llvm::DenseMap<DyLibHandle, std::string> DyLibs;
    ///\brief DynamicLibraries loaded by this Interpreter.
    ///
    DyLibs m_DyLibs;

    const InvocationOptions& m_Opts;

  public:
    ///\brief Describes the result of loading a library.
    ///
    enum LoadLibResult {
      kLoadLibSuccess, // library loaded successfully
      kLoadLibExists,  // library was already loaded
      kLoadLibError, // library was not found
      kLoadLibNumResults
    };

    DynamicLibraryManager(const InvocationOptions& Opts);
    ~DynamicLibraryManager();

    ///\brief Try to load a library file via the llvm::Linker.
    ///
    LoadLibResult tryLinker(const std::string& filename, bool permanent,
                            bool isAbsolute, bool& exists, bool& isDyLib);

    ///\brief Loads a shared library.
    ///
    ///\param [in] filename - The file to loaded.
    ///\param [in] permanent - If false, the file can be unloaded later.
    ///\param [out] tryCode - If not NULL, it will be set to false if this file
    ///        cannot be included.
    ///
    ///\returns kLoadLibSuccess on success, kLoadLibExists if the library was
    /// already loaded, kLoadLibError if the library cannot be found or any
    /// other error was encountered.
    ///
    LoadLibResult loadLibrary(const std::string& filename, bool permanent,
                              bool *tryCode = 0);

    ///\brief Returns true if the file was a dynamic library and it was already
    /// loaded.
    ///
    bool isDynamicLibraryLoaded(llvm::StringRef fullPath) const;

    ///\brief Explicitly tell the execution engine to use symbols from
    ///       a shared library that would otherwise not be used for symbol
    ///       resolution, e.g. because it was dlopened with RTLD_LOCAL.
    ///\param [in] handle - the system specific shared library handle.
    ///
    static void ExposeHiddenSharedLibrarySymbols(void* handle);
  };
} // end namespace cling
#endif // CLING_DYNAMIC_LIBRARY_MANAGER_H
