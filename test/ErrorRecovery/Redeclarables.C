// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

// Test the removal of decls from the redeclaration chain, which are marked as
// redeclarables.

extern int my_int;
.rawInput 1
int my_funct();
.rawInput 0

.storeState "testRedeclarables"
#include "Redeclarables.h"
.compareState "testRedeclarables"
// CHECK-NOT: File with AST differencies stored in: testRedeclarablesAST.diff

.rawInput 1
int my_funct() { 
  return 20;
}
.rawInput 0

int my_int = 20;

my_int
// CHECK: (int) 20

my_funct()
// CHECK: (int) 20

.q
