//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Manasij Mukherjee  <manasij7479@gmail.com>
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_AUTOLOADING_TRANSFORM_H
#define CLING_AUTOLOADING_TRANSFORM_H

#include "TransactionTransformer.h"

namespace clang {
  class Sema;
}

namespace cling {

  class AutoloadingTransform : public TransactionTransformer {
  public:
    ///\ brief Constructs the auto synthesizer.
    ///
    ///\param[in] S - The semantic analysis object.
    ///
    AutoloadingTransform(clang::Sema* S) : TransactionTransformer(S) {}

    virtual void Transform();
  };

} // namespace cling

#endif //CLING_AUTOLOADING_TRANSFORM_H
