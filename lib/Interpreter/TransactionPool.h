//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_TRANSACTION_POOL_H
#define CLING_TRANSACTION_POOL_H

#include "cling/Interpreter/Transaction.h"

#include "llvm/ADT/SmallVector.h"

namespace cling {
  class TransactionPool {
#define TRANSACTIONS_IN_BLOCK 8
  private:
    // It is twice the size of the block because there might be easily around 8
    // transactions in flight which can be empty, which might lead to refill of
    // the smallvector and then the return for reuse will exceed the capacity
    // of the smallvector causing redundant copy of the elements.
    //
    llvm::SmallVector<Transaction*, 2 * TRANSACTIONS_IN_BLOCK>  m_Transactions;

    // We need to free them in blocks.
    //
    //llvm::SmallVector<Transaction*, 64> m_TransactionBlocks;

  private:
    void RefillPool() {
      // Allocate them in one block, containing 8 transactions.
      //Transaction* arrayStart = new Transaction[TRANSACTIONS_IN_BLOCK]();
      for (size_t i = 0; i < TRANSACTIONS_IN_BLOCK; ++i)
        m_Transactions.push_back(new Transaction());
      //m_TransactionBlocks.push_back(arrayStart);
    }

  public:
    TransactionPool() {
      RefillPool();
    }

    ~TransactionPool() {
      for (size_t i = 0, e = m_Transactions.size(); i < e; ++i)
        delete m_Transactions[i];
    }

    Transaction* takeTransaction() {
      if (m_Transactions.size() == 0)
        RefillPool();
      return m_Transactions.pop_back_val();
    }

    void releaseTransaction(Transaction* T) {
      assert(T->empty() && "Transaction must be empty!");
      assert(T->getState() == Transaction::kCompleted
             && "Transaction must completed!");
      T->reset();
      m_Transactions.push_back(T);
    }
#undef TRANSACTIONS_IN_BLOCK
  };

} // end namespace cling

#endif // CLING_TRANSACTION_POOL_H
