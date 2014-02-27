//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_AST_NODE_ERASER
#define CLING_AST_NODE_ERASER

namespace llvm {
  class ExecutionEngine;
}

namespace clang {
  class Decl;
  class Sema;
}

namespace cling {

  class Transaction;

  ///\brief A simple eraser class that removes already created AST Nodes.
  ///
  class ASTNodeEraser {
  private:
    clang::Sema* m_Sema;
    llvm::ExecutionEngine* m_EEngine;

  public:
    ASTNodeEraser(clang::Sema* S, llvm::ExecutionEngine* EE);
    ~ASTNodeEraser();

    ///\brief Rolls back given transaction from the AST.
    ///
    /// Removing includes reseting various internal stuctures in the compiler to
    /// their previous states. For example it resets the lookup tables if the
    /// declaration has name and can be looked up; Reverts the redeclaration
    /// chain if the declaration was redeclarable and so on.
    /// Note1: that the code generated for the declaration is not removed yet.
    /// Note2: does not do dependency analysis.
    ///
    ///\param[in] T - The transaction to be removed.
    ///\returns true on success.
    ///
    bool RevertTransaction(Transaction* T);

    ///\brief Reverts a single decl. It must not be in any other transaction.
    /// This doesn't do dependency tracking. Use with caution.
    ///
    ///\param[in] D - The decl to be removed.
    ///
    ///\returns true on success
    ///
    bool RevertDecl(clang::Decl* D);
  };
} // end namespace cling

#endif // CLING_AST_NODE_ERASER
