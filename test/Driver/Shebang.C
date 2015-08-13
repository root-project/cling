#!/bin/cling
//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling %s 2>&1 | FileCheck %s

//CHECK-NOT: {{.*error|warning|note:.*}}
float shebang = 1.0 //CHECK: (float) 1
extern "C" int printf(const char* fmt, ...);
if(shebang == 1.0) {
  printf("I am executed\n"); // CHECK: I am executed
}
.q
