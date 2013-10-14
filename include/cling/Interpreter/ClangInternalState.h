//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: 8a2c4a806b2df22b9d375b228b1ccb502010a74f $
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_CLANG_INTERNAL_STATE_H
#define CLING_CLANG_INTERNAL_STATE_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace clang {
  class ASTContext;
  class SourceManager;
}

namespace llvm {
  class Module;
  class raw_fd_ostream;
  class raw_ostream;
}

namespace cling {
  ///\brief A helper class that stores the 'current' state of the underlying
  /// compiler - clang. It can be used for comparison of states 'before' and
  /// 'after' an event happened.
  ///
  class ClangInternalState {
  private:
    llvm::OwningPtr<llvm::raw_fd_ostream> m_LookupTablesOS;
    llvm::OwningPtr<llvm::raw_fd_ostream> m_IncludedFilesOS;
    llvm::OwningPtr<llvm::raw_fd_ostream> m_ASTOS;
    llvm::OwningPtr<llvm::raw_fd_ostream> m_LLVMModuleOS;
    std::string m_LookupTablesFile;
    std::string m_IncludedFilesFile;
    std::string m_ASTFile;
    std::string m_LLVMModuleFile;
    clang::ASTContext& m_ASTContext;
    llvm::Module& m_Module;
    std::string m_DiffCommand;
    std::string m_Name;
  public:
    ClangInternalState(clang::ASTContext& C, llvm::Module& M,
                       const std::string& Name);
    ~ClangInternalState();

    ///\brief It is convenient the state object to be named so that can be
    /// easily referenced in case of multiple.
    ///
    const std::string& getName() const { return m_Name; }

    ///\brief Stores all internal structures of the compiler into a stream.
    ///
    void store();

    ///\brief Compares two states.
    ///
    void compare(ClangInternalState& other);

    ///\brief Runs diff on two files.
    ///\param[in] file1 - A file to diff
    ///\param[in] file2 - A file to diff
    ///\param[in] differences - The differences if any between file1 and file2
    ///\returns true if there is difference in the contents.
    ///
    bool differentContent(const std::string& file1, const std::string& file2, 
                          std::string& differences, 
                   const llvm::SmallVectorImpl<const char*>* ignores = 0) const;
    
    static void printLookupTables(llvm::raw_ostream& Out, clang::ASTContext& C);
    static void printIncludedFiles(llvm::raw_ostream& Out, 
                                   clang::SourceManager& SM);
    static void printAST(llvm::raw_ostream& Out, clang::ASTContext& C);
    static void printLLVMModule(llvm::raw_ostream& Out, llvm::Module& M);
  private:
    llvm::raw_fd_ostream* createOutputFile(llvm::StringRef OutFile,
                                           std::string* TempPathName = 0,
                                           bool RemoveFileOnSignal = true);
  };
} // end namespace cling
#endif // CLING_CLANG_INTERNAL_STATE_H
