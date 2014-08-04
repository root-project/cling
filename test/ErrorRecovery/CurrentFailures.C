//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p 2>&1 | FileCheck %s
// XFAIL: *

.storeState "testCurrentFailures"

#include "Overloads.h"
error_here;

.compareState "testCurrentFailures"
// CHECK-NOT: Differences

 // This is broken case where we want to declare a function inside a wrapper
 // function, when the error recovery kicks in it segfaults.
double sin(double);
// Make FileCheck happy with having at least one positive rule:
int a = 5
// CHECK: (int) 5
.q
