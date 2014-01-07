//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Baozeng Ding
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IRDumper.h"
#include "cling/Interpreter/Transaction.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace cling {

  // pin the vtable to this file
  IRDumper::~IRDumper() {}


  void IRDumper::Transform() {
    if (!getTransaction()->getCompilationOpts().IRDebug)
      return;

    // FIXME: Size might change in the loop!
    printIR(getTransaction()->getModule());
  }

  void IRDumper::printIR(llvm::Module* M) {
    assert(M && "Must have a module to print.");
    llvm::errs() << "\n-------------------IR---------------------\n";
    M->dump();
    llvm::errs() << "\n---------------------------------------------------\n";
  }
} // namespace cling
