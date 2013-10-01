// RUN: cat %s | %cling -I%p 2>&1 | FileCheck %s
// XFAIL: *

.storeState "testCurrentFailures"

#include "Overloads.h"
error_here; 

.compareState "testCurrentFailures"
// CHECK-NOT: File with AST differencies stored in: testCurrentFailuresAST.diff

 // This is broken case where we want to declare a function inside a wrapper 
 // function, when the error recovery kicks in it segfaults.
double sin(double);
// Make FileCheck happy with having at least one positive rule: 
int a = 5
// CHECK: (int) 5
.q
