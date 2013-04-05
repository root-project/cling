//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_TRANSACTION_H
#define CLING_TRANSACTION_H

#include "cling/Interpreter/CompilationOptions.h"

#include "clang/AST/DeclGroup.h"

#include "llvm/ADT/SmallVector.h"

namespace clang {
  class Decl;
  class FunctionDecl;
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
  public:
    enum ConsumerCallInfo {
      kCCINone,
      kCCIHandleTopLevelDecl,
      kCCIHandleInterestingDecl,
      kCCIHandleTagDeclDefinition,
      kCCIHandleVTable,
      kCCIHandleCXXImplicitFunctionInstantiation,
      kCCIHandleCXXStaticMemberVarInstantiation
    };

    ///\brief Each declaration group came through different interface at 
    /// different time. We are being conservative and we want to keep all the 
    /// call sequence that originally occurred in clang.
    ///
    struct DelayCallInfo {
      clang::DeclGroupRef m_DGR;
      ConsumerCallInfo m_Call;
      DelayCallInfo(clang::DeclGroupRef DGR, ConsumerCallInfo CCI)
        : m_DGR(DGR), m_Call(CCI) {}
    };

  private:
    // Intentionally use struct instead of pair because we don't need default 
    // init.
    typedef llvm::SmallVector<DelayCallInfo, 64> DeclQueue;
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

    unsigned m_State : 3;

    unsigned m_IssuedDiags : 2;

    ///\brief Options controlling the transformers and code generator.
    ///
    CompilationOptions m_Opts;

    ///\brief The llvm Module containing the information that we will revert
    ///
    llvm::Module* m_Module;

    ///\brief The wrapper function produced by the intepreter if any.
    ///
    clang::FunctionDecl* m_WrapperFD;

    ///\brief Next transaction in if any.
    const Transaction* m_Next;

  protected:

    ///\brief Sets the next transaction in the list.
    ///
    void setNext(Transaction* T) { m_Next = T; }

  public:

    Transaction(const CompilationOptions& Opts, llvm::Module* M)
      : m_Completed(false), m_Parent(0), m_State(kCollecting),
        m_IssuedDiags(kNone), m_Opts(Opts), m_Module(M), m_WrapperFD(0),
        m_Next(0)
    { }

    ~Transaction();

    enum State {
      kCollecting,
      kRolledBack,
      kRolledBackWithErrors,
      kCommitting,
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
    typedef DeclQueue::iterator iterator;
    typedef DeclQueue::const_reverse_iterator const_reverse_iterator;
    const_iterator decls_begin() const { return m_DeclQueue.begin(); }
    iterator decls_begin() { return m_DeclQueue.begin(); }
    const_iterator decls_end() const { return m_DeclQueue.end(); }
    iterator decls_end() { return m_DeclQueue.end(); }
    const_reverse_iterator rdecls_begin() const { return m_DeclQueue.rbegin(); }
    const_reverse_iterator rdecls_end() const { return m_DeclQueue.rend(); }

    typedef NestedTransactions::const_iterator const_nested_iterator;
    typedef NestedTransactions::const_reverse_iterator const_reverse_nested_iterator;
    // FIXME: misnomer, doesn't iterate over decls!
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
        return m_DeclQueue.front().m_DGR;
      return clang::DeclGroupRef();
    }

    ///\brief Returns the last declaration of a completed transaction.
    ///
    clang::DeclGroupRef getLastDecl() const {
      if (!empty() && isCompleted())
        return m_DeclQueue.back().m_DGR;
      return clang::DeclGroupRef();
    }

    ///\brief Returns the current last transaction. Useful when the transaction
    /// in still incomplete.
    ///
    clang::DeclGroupRef getCurrentLastDecl() const {
      if (!empty())
        return m_DeclQueue.back().m_DGR;
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
      nested->setParent(this);
      // Leave a marker in the parent transaction, where the nested transaction
      // started. Using empty DeclGroupRef is save because append() filters
      // out possible empty DeclGroupRefs.      
      m_DeclQueue.push_back(DelayCallInfo(clang::DeclGroupRef(), 
                                          Transaction::kCCINone));

      m_NestedTransactions.push_back(nested);
    }

    ///\brief Direct access.
    ///
    const DelayCallInfo& operator[](size_t I) const { return m_DeclQueue[I]; }

    ///\brief Direct access, non-const.
    ///
    DelayCallInfo& operator[](size_t I) { return m_DeclQueue[I]; }

    ///\brief Returns the declaration count.
    ///
    size_t size() const { return m_DeclQueue.size(); }

    ///\brief Returns whether there are declarations in the transaction.
    ///
    bool empty() const { return m_DeclQueue.empty(); }

    ///\brief Appends a declaration group and source from which consumer interface it
    /// came from to the transaction.
    ///
    void append(DelayCallInfo DCI);

    ///\brief Appends the declaration group to the transaction as if it was 
    /// seen through HandleTopLevelDecl.
    ///
    void append(clang::DeclGroupRef DGR);

    ///\brief Wraps the declaration into declaration group and appends it to 
    /// the transaction as if it was seen through HandleTopLevelDecl.
    ///
    void append(clang::Decl* D);

    ///\brief Clears all declarations in the transaction.
    ///
    void clear() { 
      m_DeclQueue.clear(); 
      m_NestedTransactions.clear();
    }

    llvm::Module* getModule() const { return m_Module; }

    clang::FunctionDecl* getWrapperFD() const { return m_WrapperFD; }

    const Transaction* getNext() const { return m_Next; }

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

    ///\brief Prints the transaction and all subtransactions recursivly
    /// without printing any decls.
    ///
    void printStructure(size_t indent = 0) const;

    friend class IncrementalParser;
  };
} // end namespace cling

#endif // CLING_TRANSACTION_H
