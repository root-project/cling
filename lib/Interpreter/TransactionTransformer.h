//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_TRANSACTION_TRANSFORMER_H
#define CLING_TRANSACTION_TRANSFORMER_H

namespace clang {
  class Sema;
}

namespace cling {

  class Transaction;

  ///\brief Inherit from that class if you want to change/analyse declarations
  /// from the last input before code is generated.
  ///
  class TransactionTransformer {
  protected:
    clang::Sema* m_Sema;

    ///\brief Transaction being transformed.
    Transaction* m_Transaction;

  public:
    ///\brief Initializes a new transaction transformer.
    ///
    ///\param[in] S - The semantic analysis object.
    ///
    TransactionTransformer(clang::Sema* S): m_Sema(S), m_Transaction(0) {}
    virtual ~TransactionTransformer();

    ///\brief Retrieves a pointer to the semantic analysis object used for this
    /// transaction transform.
    ///
    clang::Sema* getSemaPtr() const { return m_Sema; }

    ///\brief Retreives the transaction being currently transformed.
    ///
    Transaction* getTransaction() { return m_Transaction; }

    ///\brief Retreives the transaction being currently transformed.
    ///
    const Transaction* getTransaction() const { return m_Transaction; }

    void setTransaction(Transaction* T) { m_Transaction = T; }

    ///\brief The method that does the transformation of a transaction into 
    /// another. If forwards to the protected virtual Transform method, which
    /// does the actual transformation.
    ///
    ///\param[in] T - The transaction to be transformed.
    ///\returns true on success.
    ///
    bool TransformTransaction(Transaction& T);

  protected:
    ///\brief Transforms the current transaction.
    ///
    /// Subclasses may override it in order to provide the needed behavior.
    ///
    virtual void Transform() = 0;
  };
} // end namespace cling
#endif // CLING_TRANSACTION_TRANSFORMER_H
