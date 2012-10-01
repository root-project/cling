//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_TRANSACTION_H
#define CLING_TRANSACTION_H

#include "CompilationOptions.h"

#include "clang/AST/DeclGroup.h"

#include "llvm/ADT/SmallVector.h"

namespace clang {
  class Decl;
  struct PrintingPolicy;
}

namespace llvm {
  class raw_ostream;
  class Module;
}

namespace cling {
  ///\brief Contains information about the consumed input at once.
  ///
  /// A transaction could be:
  /// - transformed - some declarations in the transaction could be modified, 
  /// deleted or some new declarations could be added.
  /// - rolled back - the declarations of the transactions could be reverted so
  /// that they weren't seen at all.
  /// - committed - code could be produced for the contents of the transaction.
  ///
  class Transaction {
  private:

    typedef llvm::SmallVector<clang::DeclGroupRef, 64> DeclQueue;
    typedef llvm::SmallVector<Transaction*, 2> NestedTransactions;

    ///\brief All seen declarations. If we collect the declarations by walking
    /// the clang::DeclContext we will miss the injected onces (eg. template 
    /// instantiations).
    ///
    DeclQueue m_DeclQueue;

    ///\brief Shows whether the transaction was commited or not.
    ///
    bool m_Completed;

    ///\brief The enclosing transaction if nested. 
    ///
    Transaction* m_Parent;

    ///\brief List of nested transactions if any.
    ///
    NestedTransactions m_NestedTransactions;

    unsigned m_State : 2;

    unsigned m_IssuedDiags : 2;

    ///\brief Options controlling the transformers and code generator.
    ///
    CompilationOptions m_Opts;

    ///\brief The llvm Module containing the information that we will revert
    ///
    llvm::Module* m_Module;

  public:

    Transaction(const CompilationOptions& Opts, llvm::Module* M)
      : m_Completed(false), m_Parent(0), m_State(kUnknown), m_IssuedDiags(kNone),
        m_Opts(Opts), m_Module(M) 
    { }

    ~Transaction();

    enum State {
      kUnknown,
      kRolledBack,
      kRolledBackWithErrors,
      kCommitted
    };

    enum IssuedDiags {
      kErrors,
      kWarnings,
      kNone
    };

    /// \{
    /// \name Iteration

    typedef DeclQueue::const_iterator const_iterator;
    typedef DeclQueue::const_reverse_iterator const_reverse_iterator;
    const_iterator decls_begin() const { return m_DeclQueue.begin(); }
    const_iterator decls_end() const { return m_DeclQueue.end(); }
    const_reverse_iterator rdecls_begin() const { return m_DeclQueue.rbegin(); }
    const_reverse_iterator rdecls_end() const { return m_DeclQueue.rend(); }

    typedef NestedTransactions::const_iterator const_nested_iterator;
    typedef NestedTransactions::const_reverse_iterator const_reverse_nested_iterator;
    const_nested_iterator nested_decls_begin() const {
      return m_NestedTransactions.begin();
    }
    const_nested_iterator nested_decls_end() const {
      return m_NestedTransactions.end();
    }
    const_reverse_nested_iterator rnested_decls_begin() const {
      return m_NestedTransactions.rbegin();
    }
    const_reverse_nested_iterator rnested_decls_end() const {
      return m_NestedTransactions.rend();
    }

    /// \}

    State getState() const { return static_cast<State>(m_State); }
    void setState(State val) { m_State = val; }

    IssuedDiags getIssuedDiags() const { 
      return static_cast<IssuedDiags>(m_IssuedDiags); 
    }
    void setIssuedDiags(IssuedDiags val) { m_IssuedDiags = val; }

    const CompilationOptions& getCompilationOpts() const { return m_Opts; }

    ///\brief Returns the first declaration of the transaction.
    ///
    clang::DeclGroupRef getFirstDecl() const {
      if (!empty())
        return m_DeclQueue.front();
      return clang::DeclGroupRef();
    }

    ///\brief Returns the last declaration of a completed transaction.
    ///
    clang::DeclGroupRef getLastDecl() const {
      if (!empty() && isCompleted())
        return m_DeclQueue.back();
      return clang::DeclGroupRef();
    }

    ///\brief Returns the current last transaction. Useful when the transaction
    /// in still incomplete.
    ///
    clang::DeclGroupRef getCurrentLastDecl() const {
      if (!empty())
        return m_DeclQueue.back();
      return clang::DeclGroupRef();
    }

    ///\ brief Returns whether the transaction is complete or not.
    ///
    /// We assume that when the last declaration of the transaction is set,
    /// the transaction is completed.
    ///
    bool isCompleted() const { return m_Completed; }
    void setCompleted(bool val = true) { m_Completed = val; }

    ///\brief If the transaction was nested into another transaction returns
    /// the parent.
    ///
    Transaction* getParent() { return m_Parent; }

    ///\brief If the transaction was nested into another transaction returns
    /// the parent.
    ///
    const Transaction* getParent() const { return m_Parent; }

    ///\brief Sets the nesting transaction of a nested transaction.
    ///
    ///\param[in] parent - The nesting transaction.
    ///
    void setParent(Transaction* parent) { m_Parent = parent; }

    bool isNestedTransaction() { return m_Parent; }
    bool hasNestedTransactions() const { return !m_NestedTransactions.empty(); }

    ///\brief Adds nested transaction to the transaction.
    ///
    ///\param[in] nested - The transaction to be nested.
    ///
    void addNestedTransaction(Transaction* nested) {
      nested->setParent(nested);
      // Leave a marker in the parent transaction, where the nested transaction
      // started. Using empty DeclGroupRef is save because append() filters
      // out possible empty DeclGroupRefs.
      m_DeclQueue.push_back(clang::DeclGroupRef());

      m_NestedTransactions.push_back(nested);
    }

    ///\brief Returns the declaration count.
    ///
    size_t size() const { return m_DeclQueue.size(); }

    ///\brief Returns whether there are declarations in the transaction.
    ///
    bool empty() const { return m_DeclQueue.empty(); }

    ///\brief Appends a declaration group to the transaction if doesn't exist.
    ///
    void appendUnique(clang::DeclGroupRef DGR);

    ///\brief Clears all declarations in the transaction.
    ///
    void clear() { 
      m_DeclQueue.clear(); 
      m_NestedTransactions.clear();
    }

    llvm::Module* getModule() const { return m_Module; }

    ///\brief Prints out all the declarations in the transaction.
    ///
    void dump() const;

    ///\brief Pretty prints out all the declarations in the transaction.
    ///
    void dumpPretty() const;

    ///\brief Customizable printout of all the declarations in the transaction.
    ///
    void print(llvm::raw_ostream& Out, const clang::PrintingPolicy& Policy,
               unsigned Indent = 0, bool PrintInstantiation = false) const;

  };
} // end namespace cling

#endif // CLING_TRANSACTION_H
