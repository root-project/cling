//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling %s "args(42)" -Xclang -verify 2>&1 | FileCheck %s

//CHECK-NOT: {{.*error|warning|note:.*}}

extern "C" int printf(const char* fmt, ...);
void args(int I, const char* S = "ArgString") {
   printf("I=%d\n", I); // CHECK: I=42
   printf("S=%s\n", S); // CHECK: S=ArgString
}
// expected-no-diagnostics
