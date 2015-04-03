//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

//RUN: %cling %s 'args(42,"AAA)BBB")' 2>&1 | FileCheck %s
//RUN: %cling '.x %s(42,"AAA)BBB")' 2>&1 | FileCheck -check-prefix=CHECK-DOTX %s

// From .x-implicit args() call:
//CHECK: input_line_4:2:2: error: no matching function for call to 'args'
//CHECK-DOTX-NOT: {{.*error|note:.*}}

extern "C" int printf(const char* fmt, ...);

void args(int I, const char* R,  const char* S = "ArgString") {

   printf("I=%d\n", I); // CHECK: I=42
   // CHECK-DOTX: I=42
   printf("S=%s\n", S); // CHECK: S=ArgString
   // CHECK-DOTX: S=ArgString
   printf("R=%s\n", R); // CHECK: R=AAA)BBB
   // CHECK-DOTX: R=AAA)BBB
}
