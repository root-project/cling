//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_AST_TRANSFORMER_H
#define CLING_AST_TRANSFORMER_H

#include "llvm/ADT/PointerIntPair.h"

#include "clang/AST/Decl.h" // for Result(Decl)
#include "clang/AST/DeclGroup.h"

#include "cling/Interpreter/Transaction.h"

namespace clang {
  class ASTConsumer;
  class Decl;
  class DeclGroupRef;
  class Sema;
}

namespace cling {

  class CompilationOptions;

  ///\brief Inherit from that class if you want to change/analyse declarations
  /// from the last input before code is generated.
  ///
  class ASTTransformer {
  protected:
    clang::Sema* m_Sema;

  private:
    clang::ASTConsumer* m_Consumer;
    Transaction* m_Transaction;

  public:
    typedef llvm::PointerIntPair<clang::Decl*, 1, bool> Result;
    ///\brief Initializes a new transaction transformer.
    ///
    ///\param[in] S - The semantic analysis object.
    ///
    ASTTransformer(clang::Sema* S):
      m_Sema(S), m_Consumer(0), m_Transaction(nullptr) {}
    virtual ~ASTTransformer();

    ///\brief Retrieves the semantic analysis object used for this transaction
    /// transform.
    ///
    clang::Sema* getSemaPtr() const { return m_Sema; }

    ///\brief Set the ASTConsumer.
    void SetConsumer(clang::ASTConsumer* Consumer) { m_Consumer = Consumer; }

    ///\brief Retrieves the current transaction.
    ///
    Transaction* getTransaction() const { return m_Transaction; }

    ///\brief Retrieves the current compilation options.
    ///
    CompilationOptions getCompilationOpts() const {
      return m_Transaction->getCompilationOpts();
    }

    ///\brief Retrieves the current compilation options (non-const).
    ///
    CompilationOptions& getCompilationOpts() {
      return m_Transaction->getCompilationOpts();
    }

    ///\brief Emit declarations that are created during the transformation.
    ///
    ///\param[in] DGR - Decls to be emitted.
    ///
    void Emit(clang::DeclGroupRef DGR);

    ///\brief Emit a declaration that is created during the transformation.
    ///
    ///\param[in] D - Decl to be emitted.
    ///
    void Emit(clang::Decl* D) { Emit(clang::DeclGroupRef(D)); }

    ///\brief Transforms the declaration, forward to Transform(D).
    ///
    ///\param[in] D - The declaration to be transformed.
    ///\param[in] T - The declaration's transaction.
    ///\returns The transformation result which will be emitted. Return nullptr
    ///  if this declaration should not be emitted. Returning error will abort
    ///  the transaction.
    ///
    Result Transform(clang::Decl* D, Transaction* T) {
      m_Transaction = T;
      return Transform(D);
    }

  protected:
    ///\brief Transforms the declaration.
    ///
    /// Subclasses override it in order to provide the needed behavior.
    ///
    ///\param[in] D - The declaration to be transformed.
    ///\returns The transformation result which will be emitted. Return nullptr
    ///  if this declaration should not be emitted. Returning error will abort
    ///  the transaction.
    ///
    virtual Result Transform(clang::Decl* D) = 0;

  };

  class WrapperTransformer: public ASTTransformer {
  public:
    WrapperTransformer(clang::Sema* S): ASTTransformer(S) {}
  };
} // end namespace cling
#endif // CLING_AST_TRANSFORMER_H
