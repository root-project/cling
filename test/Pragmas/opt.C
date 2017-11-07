//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

extern "C" int printf(const char*,...);
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"

gCling->getDefaultOptLevel()
// CHECK: (int) 0

(int)gCling->getLatestTransaction()->getCompilationOpts().OptLevel
// CHECK-NEXT: (int) 0

{
#pragma cling optimize(0)
  printf("Transaction OptLevel=%d\n", (int)gCling->getLatestTransaction()->getCompilationOpts().OptLevel);
}
// CHECK-NEXT: Transaction OptLevel=0

{
#pragma cling optimize(1)
  printf("Transaction OptLevel=%d\n", (int)gCling->getLatestTransaction()->getCompilationOpts().OptLevel);
}
// CHECK-NEXT: Transaction OptLevel=1

{
#pragma cling optimize(2)
  printf("Transaction OptLevel=%d\n", (int)gCling->getLatestTransaction()->getCompilationOpts().OptLevel);
}
// CHECK-NEXT: Transaction OptLevel=2

{
#pragma cling optimize(2)
// CHECK-NEXT: cling::PHOptLevel: conflicting `#pragma cling optimize` directives: was already set to 2
#pragma cling optimize(1)
// CHECK-NEXT: Setting to lower value of 1
#pragma cling optimize(2)
// CHECK-NEXT: cling::PHOptLevel: conflicting `#pragma cling optimize` directives: was already set to 1
// CHECK-NEXT: Ignoring higher value of 2
  printf("Transaction OptLevel=%d\n", (int)gCling->getLatestTransaction()->getCompilationOpts().OptLevel);
}
// CHECK-NEXT: Transaction OptLevel=1

.O
// CHECK-NEXT: Current cling optimization level: 0

// No parenthesis
{
#pragma cling optimize 1
  printf("Transaction OptLevel=%d\n", (int)gCling->getLatestTransaction()->getCompilationOpts().OptLevel);
}
// CHECK-NEXT: Transaction OptLevel=1

// Full functional style
{
#pragma cling(optimize, 3)
  printf("Transaction OptLevel=%d\n", (int)gCling->getLatestTransaction()->getCompilationOpts().OptLevel);
}
// CHECK-NEXT: Transaction OptLevel=3

// Invalid token
#pragma cling optimize Invalid // expected-error {{expected argument after optimize}}

.q
