// RUN: cat %s | %cling -I%p 2>&1 | FileCheck %s
// XFAIL: *

.storeState "testCurrentFailures"

#include "Overloads.h"
error_here; 

.compareState "testCurrentFailures"
// CHECK-NOT: File with AST differencies stored in: testCurrentFailuresAST.diff
// Make FileCheck happy with having at least one positive rule: 
int a = 5
// CHECK: (int) 5
.q
