//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -noruntime -Xclang -verify 2>&1 | FileCheck %s
// Test noruntimeTest

extern "C" int printf(const char*,...);
int TEST = 9;
TEST
// CHECK-NOT: (int) 9

printf("TEST: %d\n", TEST);
// CHECK: TEST: 9

// expected-no-diagnostics
.q
