//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Baozeng Ding
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_IR_DUMPER_H
#define CLING_IR_DUMPER_H

#include "TransactionTransformer.h"

namespace llvm {
  class Module;
}

namespace cling {

  class Transaction;

  // TODO : This is not really a transformer. Factor out.
  class IRDumper : public TransactionTransformer {
  public:
    IRDumper() : TransactionTransformer(0) { }
    virtual ~IRDumper();

    virtual void Transform();

  private:
    void printIR(llvm::Module* M);
  };

} // namespace cling

#endif // CLING_IR_DUMPER_H
