//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_DECL_COLLECTOR_H
#define CLING_DECL_COLLECTOR_H

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/Lex/PPCallbacks.h"

#include "ASTTransformer.h"

#include <vector>
#include <memory>

namespace clang {
  class ASTContext;
  class CodeGenerator;
  class Decl;
  class DeclGroupRef;
  class Token;
}

namespace cling {

  class ASTTransformer;
  class WrapperTransformer;
  class DeclCollector;
  class IncrementalParser;
  class Transaction;

  ///\brief Serves as DeclCollector's connector to the PPCallbacks interface.
  ///
  class DeclCollectorPPAdapter: public clang::PPCallbacks {
    DeclCollector* m_parent;
  public:
    DeclCollectorPPAdapter(DeclCollector* parent):
      m_parent(parent)
    {}

    /// \name PPCallbacks overrides
    /// Macro support
    void MacroDefined(const clang::Token &MacroNameTok,
                      const clang::MacroDirective *MD) override;
    /// \}
  };

  ///\brief Collects declarations and fills them in cling::Transaction.
  ///
  /// cling::Transaction becomes is a main building block in the interpreter.
  /// cling::DeclCollector is responsible for appending all the declarations
  /// seen by clang.
  ///
  class DeclCollector: public clang::ASTMutationListener,
                       public clang::ASTConsumer  {
  private:
    ///\brief Contains the transaction AST transformers.
    ///
    std::vector<std::unique_ptr<ASTTransformer>> m_TransactionTransformers;

    ///\brief Contains the AST transformers operating on the wrapper.
    ///
    std::vector<std::unique_ptr<WrapperTransformer>> m_WrapperTransformers;

    IncrementalParser* m_IncrParser;
    clang::ASTConsumer* m_Consumer;
    Transaction* m_CurTransaction;

    ///\brief Test whether the first decl of the DeclGroupRef comes from an AST
    /// file.
    ///
    bool comesFromASTReader(clang::DeclGroupRef DGR) const;
    bool comesFromASTReader(const clang::Decl* D) const;

    bool Transform(clang::DeclGroupRef& DGR) const;

    ///\brief Runs AST transformers on a transaction.
    ///
    ///\param[in] D - the decl to be transformed.
    ///
    ASTTransformer::Result TransformDecl(clang::Decl* D) const;

  public:
    DeclCollector() :
      m_IncrParser(0), m_Consumer(0), m_CurTransaction(0) {}

    virtual ~DeclCollector();

    std::unique_ptr<DeclCollectorPPAdapter> MakePPAdapter() {
      return std::unique_ptr<DeclCollectorPPAdapter>
        (new DeclCollectorPPAdapter(this));
    }

    void SetTransformers(std::vector<std::unique_ptr<ASTTransformer>>&& TT,
                         std::vector<std::unique_ptr<WrapperTransformer>>&& WT){
      m_TransactionTransformers.swap(TT);
      m_WrapperTransformers.swap(WT);
      for (auto&& TT: m_TransactionTransformers)
        TT->SetConsumer(this);
      for (auto&& WT: m_WrapperTransformers)
        WT->SetConsumer(this);
    }

    void setContext(IncrementalParser* IncrParser, ASTConsumer* Consumer) {
      m_IncrParser = IncrParser;
      m_Consumer = Consumer;
    }

    /// \name PPCallbacks overrides
    /// Macro support
    void MacroDefined(const clang::Token &MacroNameTok,
                              const clang::MacroDirective *MD);
    /// \}
    /// \name ASTMutationListeners overrides
    virtual void AddedCXXImplicitMember(const clang::CXXRecordDecl *RD,
                                        const clang::Decl *D);
    /// \}

    /// \{
    /// \name ASTConsumer overrides

    virtual bool HandleTopLevelDecl(clang::DeclGroupRef DGR);
    virtual void HandleInterestingDecl(clang::DeclGroupRef DGR);
    virtual void HandleTagDeclDefinition(clang::TagDecl* TD);
    virtual void HandleInvalidTagDeclDefinition(clang::TagDecl* TD);
    virtual void HandleVTable(clang::CXXRecordDecl* RD);
    virtual void CompleteTentativeDefinition(clang::VarDecl* VD);
    virtual void HandleTranslationUnit(clang::ASTContext& Ctx);
    virtual void HandleCXXImplicitFunctionInstantiation(clang::FunctionDecl *D);
    virtual void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *D);
    /// \}

    /// \{
    /// \name Transaction Support

    Transaction* getTransaction() { return m_CurTransaction; }
    const Transaction* getTransaction() const { return m_CurTransaction; }
    void setTransaction(Transaction* curT) { m_CurTransaction = curT; }
    void setTransaction(const Transaction* curT) {
      m_CurTransaction = const_cast<Transaction*>(curT);
    }

    /// \}

    // dyn_cast/isa support
    static bool classof(const clang::ASTConsumer*) { return true; }
  };

  inline void
  DeclCollectorPPAdapter::MacroDefined(const clang::Token &MacroNameTok,
                                       const clang::MacroDirective *MD) {
    m_parent->MacroDefined(MacroNameTok, MD);
  }

} // namespace cling

#endif // CLING_DECL_COLLECTOR_H
