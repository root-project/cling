//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_CLANG_INTERNAL_STATE_H
#define CLING_CLANG_INTERNAL_STATE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace clang {
  class ASTContext;
  class CodeGenerator;
  class SourceManager;
  class Preprocessor;
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
    std::string m_LookupTablesFile;
    std::string m_IncludedFilesFile;
    std::string m_ASTFile;
    std::string m_LLVMModuleFile;
    std::string m_MacrosFile;
    clang::ASTContext& m_ASTContext;
    clang::Preprocessor& m_Preprocessor;
    clang::CodeGenerator* m_CodeGen;
    llvm::Module* m_Module;
    std::string m_DiffCommand;
    std::string m_Name;
    ///\brief Takes the ownership after compare was made.
    ///
    std::unique_ptr<ClangInternalState> m_DiffPair;
  public:
    ClangInternalState(clang::ASTContext& AC, clang::Preprocessor& PP,
                       llvm::Module* M, clang::CodeGenerator* CG,
                       const std::string& name);
    ~ClangInternalState();

    ///\brief It is convenient the state object to be named so that can be
    /// easily referenced in case of multiple.
    ///
    const std::string& getName() const { return m_Name; }

    ///\brief Stores all internal structures of the compiler into a stream.
    ///
    void store();

    ///\brief Compares the states with the current state of the same objects.
    ///
    void compare(const std::string& name);

    ///\brief Runs diff on two files.
    ///\param[in] file1 - A file to diff
    ///\param[in] file2 - A file to diff
    ///\param[in] differences - The differences if any between file1 and file2
    ///\param[in] ignores - A list of differences to ignore.
    ///\returns true if there is difference in the contents.
    ///
    bool differentContent(const std::string& file1, const std::string& file2,
                          std::string& differences,
                   const llvm::SmallVectorImpl<const char*>* ignores = 0) const;

    static void printLookupTables(llvm::raw_ostream& Out, clang::ASTContext& C);
    static void printIncludedFiles(llvm::raw_ostream& Out,
                                   clang::SourceManager& SM);
    static void printAST(llvm::raw_ostream& Out, clang::ASTContext& C);
    static void printLLVMModule(llvm::raw_ostream& Out, llvm::Module& M,
                                clang::CodeGenerator& CG);
    static void printMacroDefinitions(llvm::raw_ostream& Out,
                                      clang::Preprocessor& PP);
  private:
    llvm::raw_fd_ostream* createOutputFile(llvm::StringRef OutFile,
                                           std::string* TempPathName = 0,
                                           bool RemoveFileOnSignal = true);
  };
} // end namespace cling
#endif // CLING_CLANG_INTERNAL_STATE_H
