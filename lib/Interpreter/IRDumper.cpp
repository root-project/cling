//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Baozeng Ding
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "IRDumper.h"
#include "cling/Interpreter/Transaction.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace cling {

  // pin the vtable to this file
  IRDumper::~IRDumper() {}


  void IRDumper::Transform() {
    if (!getTransaction()->getCompilationOpts().Debug)
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
