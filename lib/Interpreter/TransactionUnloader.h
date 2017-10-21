//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_TRANSACTION_UNLOADER
#define CLING_TRANSACTION_UNLOADER

#include <memory>

namespace llvm {
  class Module;
}

namespace clang {
  class CodeGenerator;
  class Decl;
  class Sema;
}

namespace cling {

  class IncrementalExecutor;
  class Interpreter;
  class Transaction;
  class DeclUnloader;

  ///\brief A simple eraser class that removes already created AST Nodes.
  ///
  class TransactionUnloader {
  private:
    cling::Interpreter* m_Interp;
    clang::Sema* m_Sema;
    clang::CodeGenerator* m_CodeGen;
    cling::IncrementalExecutor* m_Exe;

    bool unloadDeclarations(Transaction* T, DeclUnloader& DeclU);
    bool unloadDeserializedDeclarations(Transaction* T,
                                        DeclUnloader& DeclU);
    bool unloadFromPreprocessor(Transaction* T, DeclUnloader& DeclU);
    bool unloadModule(const std::shared_ptr<llvm::Module>& M);

  public:
    TransactionUnloader(cling::Interpreter* I, clang::Sema* Sema,
                        clang::CodeGenerator* CG,
                        cling::IncrementalExecutor* Exe):
      m_Interp(I), m_Sema(Sema), m_CodeGen(CG), m_Exe(Exe) {}

    ///\brief Rolls back given transaction from the AST.
    ///
    /// Removing includes reseting various internal stuctures in the compiler to
    /// their previous states. For example it resets the lookup tables if the
    /// declaration has name and can be looked up; Unloads the redeclaration
    /// chain if the declaration was redeclarable and so on.
    /// Note1: that the code generated for the declaration is not removed yet.
    /// Note2: does not do dependency analysis.
    ///
    ///\param[in] T - The transaction to be removed.
    ///\returns true on success.
    ///
    bool RevertTransaction(Transaction* T);

    ///\brief Unloads a single decl. It must not be in any other transaction.
    /// This doesn't do dependency tracking. Use with caution.
    ///
    ///\param[in] D - The decl to be removed.
    ///
    ///\returns true on success
    ///
    bool UnloadDecl(clang::Decl* D);

    ///\brief Get the IncrementalExecutor from which transactions should be
    /// unloaded.
    cling::IncrementalExecutor* getExecutor() const { return m_Exe; }
  };
} // end namespace cling

#endif // CLING_AST_NODE_ERASER
