//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_CHECK_EMPTY_TRANSACTION_TRANSFORMER
#define CLING_CHECK_EMPTY_TRANSACTION_TRANSFORMER

#include "TransactionTransformer.h"

namespace clang {
  class Sema;
}

namespace cling {

  class CheckEmptyTransactionTransformer : public TransactionTransformer {
  public:
    CheckEmptyTransactionTransformer(clang::Sema* S)
      : TransactionTransformer(S) { }
    virtual void Transform();
  };
} // end namespace cling

#endif // CLING_CHECK_EMPTY_TRANSACTION_TRANSFORMER
