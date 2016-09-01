//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %built_cling -fno-rtti | FileCheck %s

// This test makes sure the interpreter doesn't create many useless empty
// transactions.

// Author: Vassil Vassilev

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/AST/Decl.h"

#include <stdio.h>

using namespace cling;

void generateNestedTransaction(int depth) {
  if (!depth)
    return;
  cling::Interpreter::PushTransactionRAII RAIIT(gCling);
  if (depth | 0x1) { // if odd
    char buff[100];
    sprintf(buff, "int i%d;", depth);
    gCling->process(buff);
  } // this will cause every even transaction to be reused.
  generateNestedTransaction(--depth);
}

generateNestedTransaction(5);
const cling::Transaction* T = gCling->getFirstTransaction();
while(T) {
  if (T->empty())
    printf("Empty transaction detected!\n");
  else if (T->getWrapperFD() && T->getWrapperFD()->getKind() != clang::Decl::Function)
    printf("Unexpected wrapper kind!\n");
  if (T->getState() != Transaction::kCommitted)
    printf("Unexpected transaction state!\n");
  //T->printStructure();
  T = T->getNext();
}
printf("Just make FileCheck(CHECK-NOT) happy.\n")
//CHECK-NOT:Empty transaction detected!
//CHECK-NOT:Unexpected wrapper kind!
//CHECK-NOT:Unexpected transaction state!
.q
