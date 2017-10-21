//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_TRANSACTION_H
#define CLING_TRANSACTION_H

#include "cling/Interpreter/CompilationOptions.h"

#include "clang/AST/DeclGroup.h"
#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"

#include <memory>

namespace clang {
  class ASTContext;
  class Decl;
  class FunctionDecl;
  class IdentifierInfo;
  class NamedDecl;
  class MacroDirective;
  class Preprocessor;
  struct PrintingPolicy;
  class Sema;
}

namespace llvm {
  class raw_ostream;
}

namespace cling {
  class IncrementalExecutor;
  class TransactionPool;

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
      kCCIHandleCXXStaticMemberVarInstantiation,
      kCCICompleteTentativeDefinition,
      kCCINumStates
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
      inline bool operator==(const DelayCallInfo& rhs) {
        return m_DGR.getAsOpaquePtr() == rhs.m_DGR.getAsOpaquePtr()
          && m_Call == rhs.m_Call;
      }
      inline bool operator!=(const DelayCallInfo& rhs) {
        return !operator==(rhs);
      }
      void dump() const;
      void print(llvm::raw_ostream& Out, const clang::PrintingPolicy& Policy,
                 unsigned Indent, bool PrintInstantiation,
                 llvm::StringRef prependInfo = "") const;
    };

    ///\brief Each macro pair (is this the same as for decls?)came
    /// through different interface at
    /// different time. We are being conservative and we want to keep all the
    /// call sequence that originally occurred in clang.
    ///
    struct MacroDirectiveInfo {
      // We need to store both the IdentifierInfo and the MacroDirective
      // because the Preprocessor stores the macros in a DenseMap<II, MD>.
      clang::IdentifierInfo* m_II;
      const clang::MacroDirective* m_MD;
      MacroDirectiveInfo(clang::IdentifierInfo* II,
                         const clang::MacroDirective* MD)
                : m_II(II), m_MD(MD) {}
      inline bool operator==(const MacroDirectiveInfo& rhs) {
        return m_II == rhs.m_II && m_MD == rhs.m_MD;
      }
      inline bool operator!=(const MacroDirectiveInfo& rhs) {
        return !operator==(rhs);
      }
      void dump(const clang::Preprocessor& PP) const;
      void print(llvm::raw_ostream& Out, const clang::Preprocessor& PP) const;
    };

  private:
    // Intentionally use struct instead of pair because we don't need default
    // init.
    typedef llvm::SmallVector<DelayCallInfo, 64> DeclQueue;
    typedef llvm::SmallVector<Transaction*, 2> NestedTransactions;

    ///\brief All seen declarations, except the deserialized ones.
    /// If we collect the declarations by walking the clang::DeclContext we
    /// will miss the injected onces (eg. template instantiations).
    ///
    DeclQueue m_DeclQueue;

    ///\brief All declarations that the transaction caused to be deserialized,
    /// either from the PCH or the PCM.
    ///
    DeclQueue m_DeserializedDeclQueue;

    ///\brief List of nested transactions if any.
    ///
    std::unique_ptr<NestedTransactions> m_NestedTransactions;

    ///\brief The enclosing transaction if nested.
    ///
    Transaction* m_Parent;

    unsigned m_State : 3;

    unsigned m_IssuedDiags : 2;

    ///\brief Options controlling the transformers and code generator.
    ///
    CompilationOptions m_Opts;

    ///\brief The llvm Module containing the information that we will revert
    ///
    std::shared_ptr<llvm::Module> m_Module;

    ///\brief The Executor to use m_ExeUnload on.
    ///
    IncrementalExecutor* m_Exe;

    ///\brief The wrapper function produced by the intepreter if any.
    ///
    clang::FunctionDecl* m_WrapperFD;

    ///\brief Next transaction in if any.
    ///
    const Transaction* m_Next;

    ///\brief The Sema holding the ASTContext and the Preprocessor.
    ///
    clang::Sema& m_Sema;

    // Intentionally use struct instead of pair because we don't need default
    // init.
    // Add macro decls to be able to revert them for error recovery.
    typedef llvm::SmallVector<MacroDirectiveInfo, 2> MacroDirectiveInfoQueue;

    ///\brief All seen macros.
    ///
    MacroDirectiveInfoQueue m_MacroDirectiveInfoQueue;

    ///\brief The FileID for the top-most memory buffer that started the
    /// transaction.
    ///
    clang::FileID m_BufferFID;

    /// TransactionPool needs direct access to m_State as setState asserts
    friend class TransactionPool;

    void Initialize(clang::Sema& S);

  public:
    enum State {
      kCollecting,
      kCompleted,
      kRolledBack,
      kRolledBackWithErrors,
      kCommitted,
      kNumStates
    };

    enum IssuedDiags {
      kErrors,
      kWarnings,
      kNone
    };

    Transaction(clang::Sema& S);
    Transaction(const CompilationOptions& Opts, clang::Sema& S);
    ~Transaction();

    /// \{
    /// \name Iteration

    typedef DeclQueue::iterator iterator;
    typedef DeclQueue::const_iterator const_iterator;
    typedef DeclQueue::const_reverse_iterator const_reverse_iterator;
    iterator decls_begin() {
      return m_DeclQueue.begin();
    }
    iterator decls_end() {
      return m_DeclQueue.end();
    }
    const_iterator decls_begin() const {
      return m_DeclQueue.begin();
    }
    const_iterator decls_end() const {
      return m_DeclQueue.end();
    }
    const_reverse_iterator rdecls_begin() const {
      return m_DeclQueue.rbegin();
    }
    const_reverse_iterator rdecls_end() const {
      return m_DeclQueue.rend();
    }

    iterator deserialized_decls_begin() {
      return m_DeserializedDeclQueue.begin();
    }
    iterator deserialized_decls_end() {
      return m_DeserializedDeclQueue.end();
    }
    const_iterator deserialized_decls_begin() const {
      return m_DeserializedDeclQueue.begin();
    }
    const_iterator deserialized_decls_end() const {
      return m_DeserializedDeclQueue.end();
    }
    const_reverse_iterator deserialized_rdecls_begin() const {
      return m_DeserializedDeclQueue.rbegin();
    }
    const_reverse_iterator deserialized_rdecls_end() const {
      return m_DeserializedDeclQueue.rend();
    }

    typedef NestedTransactions::const_iterator const_nested_iterator;
    typedef NestedTransactions::const_reverse_iterator const_reverse_nested_iterator;
    const_nested_iterator nested_begin() const {
      if (hasNestedTransactions())
        return m_NestedTransactions->begin();
      return 0;
    }
    const_nested_iterator nested_end() const {
      if (hasNestedTransactions())
        return m_NestedTransactions->end();
      return 0;
    }
    const_reverse_nested_iterator rnested_begin() const {
      if (hasNestedTransactions())
        return m_NestedTransactions->rbegin();
      return const_reverse_nested_iterator(0);
    }
    const_reverse_nested_iterator rnested_end() const {
      if (hasNestedTransactions())
        return m_NestedTransactions->rend();
      return const_reverse_nested_iterator(0);
    }

    /// Macro iteration
    typedef MacroDirectiveInfoQueue::iterator macros_iterator;
    typedef MacroDirectiveInfoQueue::const_iterator const_macros_iterator;
    typedef MacroDirectiveInfoQueue::const_reverse_iterator const_reverse_macros_iterator;

    macros_iterator macros_begin() {
      return m_MacroDirectiveInfoQueue.begin();
    }
    macros_iterator macros_end() {
      return m_MacroDirectiveInfoQueue.end();
    }
    const_macros_iterator macros_begin() const {
      return m_MacroDirectiveInfoQueue.begin();
    }
    const_macros_iterator macros_end() const {
      return m_MacroDirectiveInfoQueue.end();
    }
    const_reverse_macros_iterator rmacros_begin() const {
      return m_MacroDirectiveInfoQueue.rbegin();
    }
    const_reverse_macros_iterator rmacros_end() const {
      return m_MacroDirectiveInfoQueue.rend();
    }


    /// \}

    State getState() const { return static_cast<State>(m_State); }
    void setState(State val) {
      assert(m_State != kNumStates
             && "Transaction already returned in the pool");
      m_State = val;
    }

    IssuedDiags getIssuedDiags() const {
      return static_cast<IssuedDiags>(getTopmostParent()->m_IssuedDiags);
    }
    void setIssuedDiags(IssuedDiags val) {
      getTopmostParent()->m_IssuedDiags = val;
    }

    const CompilationOptions& getCompilationOpts() const { return m_Opts; }
    CompilationOptions& getCompilationOpts() { return m_Opts; }
    void setCompilationOpts(const CompilationOptions& CO) {
      assert(getState() == kCollecting && "Something wrong with you?");
      m_Opts = CO;
    }

    ///\brief Returns the first declaration of the transaction.
    ///
    clang::DeclGroupRef getFirstDecl() const {
      if (!m_DeclQueue.empty())
        return m_DeclQueue.front().m_DGR;
      return clang::DeclGroupRef();
    }

    ///\brief Returns the last declaration of a completed transaction.
    ///
    clang::DeclGroupRef getLastDecl() const {
      if (!m_DeclQueue.empty() && isCompleted())
        return m_DeclQueue.back().m_DGR;
      return clang::DeclGroupRef();
    }

    ///\brief Returns the NamedDecl* if a Decl with name is present, 0 otherwise.
    ///
    clang::NamedDecl* containsNamedDecl(llvm::StringRef name) const;

    ///\brief Returns the current last transaction. Useful when the transaction
    /// in still incomplete.
    ///
    clang::DeclGroupRef getCurrentLastDecl() const {
      if (!m_DeclQueue.empty())
        return m_DeclQueue.back().m_DGR;
      return clang::DeclGroupRef();
    }

    ///\ brief Returns whether the transaction is complete or not.
    ///
    /// We assume that when the last declaration of the transaction is set,
    /// the transaction is completed.
    ///
    bool isCompleted() const { return m_State >= kCompleted; }

    ///\brief If the transaction was nested into another transaction returns
    /// the parent.
    ///
    Transaction* getParent() { return m_Parent; }

    ///\brief If the transaction was nested into another transaction returns
    /// the parent.
    ///
    const Transaction* getParent() const { return m_Parent; }

    ///\brief If the transaction was nested into another transaction returns
    /// the topmost transaction, else this.
    ///
    Transaction* getTopmostParent() {
      const Transaction* ConstThis = const_cast<const Transaction*>(this);
      return const_cast<Transaction*>(ConstThis->getTopmostParent());
    }

    ///\brief If the transaction was nested into another transaction returns
    /// the topmost transaction, else this.
    ///
    const Transaction* getTopmostParent() const {
      const Transaction* ret = this;
      while (ret->getParent())
        ret = ret->getParent();
      return ret;
    }

    ///\brief Sets the nesting transaction of a nested transaction.
    ///
    ///\param[in] parent - The nesting transaction.
    ///
    void setParent(Transaction* parent) { m_Parent = parent; }

    bool isNestedTransaction() const { return m_Parent; }
    bool hasNestedTransactions() const { return m_NestedTransactions.get(); }

    ///\brief Adds nested transaction to the transaction.
    ///
    ///\param[in] nested - The transaction to be nested.
    ///
    void addNestedTransaction(Transaction* nested);

    ///\brief Removes a nested transaction.
    ///
    ///\param[in] nested - The transaction to be removed.
    ///
    void removeNestedTransaction(Transaction* nested);

    Transaction* getLastNestedTransaction() const {
      if (!hasNestedTransactions())
        return 0;
      return m_NestedTransactions->back();
    }

    ///\brief Returns whether there are declarations in the transaction.
    ///
    bool empty() const {
      return m_DeclQueue.empty() && m_DeserializedDeclQueue.empty()
        && (!m_NestedTransactions || m_NestedTransactions->empty())
        && m_MacroDirectiveInfoQueue.empty();
    }

    ///\brief Appends a declaration group and source from which consumer
    /// interface it came from to the transaction.
    ///
    void append(DelayCallInfo DCI);

    ///\brief Appends a declaration group to a transaction even if it was
    /// completed and ready for codegenning.
    /// NOTE: Please use with caution!
    ///
    void forceAppend(DelayCallInfo DCI);

    ///\brief Appends the declaration group to the transaction as if it was
    /// seen through HandleTopLevelDecl.
    ///
    void append(clang::DeclGroupRef DGR);

    ///\brief Wraps the declaration into declaration group and appends it to
    /// the transaction as if it was seen through HandleTopLevelDecl.
    ///
    void append(clang::Decl* D);

    ///\brief Wraps the declaration into declaration group and appends it to
    /// the transaction as if it was seen through HandleTopLevelDecl,  even if
    /// it was completed and ready for codegenning.
    /// NOTE: Please use with caution!
    ///
    void forceAppend(clang::Decl *D);

    ///\brief Appends the declaration of a macro.
    void append(MacroDirectiveInfo MDE);

    ///\brief Clears all declarations in the transaction.
    ///
    void clear() {
      m_DeclQueue.clear();
      if (m_NestedTransactions)
        m_NestedTransactions->clear();
    }

    std::shared_ptr<llvm::Module> getModule() const { return m_Module; }
    void setModule(std::unique_ptr<llvm::Module> M) { m_Module = std::move(M); }

    IncrementalExecutor* getExecutor() const { return m_Exe; }

    clang::FunctionDecl* getWrapperFD() const { return m_WrapperFD; }

    const Transaction* getNext() const { return m_Next; }
    void setNext(Transaction* T) { m_Next = T; }

    void setBufferFID(clang::FileID FID) { m_BufferFID = FID; }
    clang::FileID getBufferFID() const { return m_BufferFID; }
    clang::SourceLocation getSourceStart(const clang::SourceManager& SM) const;

    ///\brief The transactions could be reused and the pointer couldn't serve
    /// as a unique handle to a transaction. Unique handles are used by clients
    /// which want to check whether the interpreter saw more input.
    ///
    ///\returns a unique handle to the transaction.
    ///
    unsigned getUniqueID() const;

    ///\brief Erases an element at given position.
    ///
    void erase(iterator pos);

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
    void printStructure(size_t nindent = 0) const;

    void printStructureBrief(size_t nindent = 0) const;

  private:
    bool comesFromASTReader(clang::DeclGroupRef DGR) const;
  };

} // end namespace cling

#endif // CLING_TRANSACTION_H
