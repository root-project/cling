//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#include "TransactionTransformer.h"

#include "clang/Sema/Sema.h"

namespace cling {

  // pin the vtable here since there is no point to create dedicated to that
  // cpp file.
  TransactionTransformer::~TransactionTransformer() {}

  bool TransactionTransformer::TransformTransaction(Transaction& T) {
    m_Transaction = &T;
    Transform();

    if (!m_Sema)
      return true;

    return !m_Sema->getDiagnostics().hasErrorOccurred();
  }
} // end namespace cling
