//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
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
