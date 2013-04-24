// RUN: cat %s | %cling | FileCheck %s

// This test makes sure the interpreter doesn't create many useless empty
// transactions.

// Author: Vassil Vassilev

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include <stdio.h>


.rawInput 1

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

.rawInput 0

generateNestedTransaction(5);
const cling::Transaction* T = gCling->getFirstTransaction();
while(T) {
  if (!T->size())
    printf("Empty transaction detected!\n");
  //T->printStructure();
  T = T->getNext();
}
printf("Just make FileCheck(CHECK-NOT) happy.\n")
//CHECK-NOT:Empty transaction detected!
.q
