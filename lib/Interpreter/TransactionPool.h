//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_TRANSACTION_POOL_H
#define CLING_TRANSACTION_POOL_H

#include "cling/Interpreter/Transaction.h"

#include "llvm/ADT/SmallVector.h"

namespace clang {
  class Sema;
}

namespace cling {
  class TransactionPool {
    enum {
#ifdef NDEBUG
      kDebugMode           = 0,
#else
      kDebugMode           = 1, // Always use a new Transaction
#endif
      kTransactionsInBlock = 8,
      kPoolSize            = 2 * kTransactionsInBlock
    };

    // It is twice the size of the block because there might be easily around 8
    // transactions in flight which can be empty, which might lead to refill of
    // the smallvector and then the return for reuse will exceed the capacity
    // of the smallvector causing redundant copy of the elements.
    //
    llvm::SmallVector<Transaction*, kPoolSize>  m_Transactions;

  public:
    TransactionPool() {}
    ~TransactionPool() {
      // Only free the memory as anything put in m_Transactions will have
      // already been destructed in releaseTransaction
      for (Transaction* T : m_Transactions)
        ::operator delete(T);
    }

    Transaction* takeTransaction(clang::Sema& S) {
      Transaction *T;
      if (kDebugMode || m_Transactions.empty())
        T = new Transaction(S);
      else
        T = new (m_Transactions.pop_back_val()) Transaction(S);

      return T;
    }

    // Transaction T must be from call to TransactionPool::takeTransaction
    //
    void releaseTransaction(Transaction* T, bool reuse = true) {
      if (reuse) {
        assert((T->getState() == Transaction::kCompleted ||
                T->getState() == Transaction::kRolledBack)
               && "Transaction must be completed!");
        // Force reuse to off when not in Debug mode
        if (kDebugMode)
          reuse = false;
      }

      // Tell the parent that T is gone.
      if (T->getParent())
        T->getParent()->removeNestedTransaction(T);

      // don't overflow the pool
      if (reuse && (m_Transactions.size() < kPoolSize)) {
        T->m_State = Transaction::kNumStates;
        T->~Transaction();
        m_Transactions.push_back(T);
      }
      else
       delete T;
    }
  };

} // end namespace cling

#endif // CLING_TRANSACTION_POOL_H
