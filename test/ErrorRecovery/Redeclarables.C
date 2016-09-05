//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p -Xclang -verify 2>&1 | FileCheck %s

// Test the removal of decls from the redeclaration chain, which are marked as
// redeclarables.

extern int my_int;
.rawInput 1
int my_funct();
.rawInput 0

.storeState "testRedeclarables"
#include "Redeclarables.h"
.compareState "testRedeclarables"
// CHECK-NOT: Differences

int my_funct() {
  return 20;
}

int my_int = 20;

my_int
// CHECK: (int) 20

my_funct()
// CHECK: (int) 20

.q
