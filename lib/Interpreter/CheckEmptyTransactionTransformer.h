//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: a88f59662c0204c76ab2204361d9f64b1e10fcfc $
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_CHECK_EMPTY_TRANSACTION_TRANSFORMER
#define CLING_CHECK_EMPTY_TRANSACTION_TRANSFORMER

#include "TransactionTransformer.h"

namespace cling {

  class CheckEmptyTransactionTransformer : public TransactionTransformer {
  public:
    CheckEmptyTransactionTransformer() : TransactionTransformer(/*Sema=*/0) { }
    virtual void Transform();
  };
} // end namespace cling

#endif // CLING_CHECK_EMPTY_TRANSACTION_TRANSFORMER
