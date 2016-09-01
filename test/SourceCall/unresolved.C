//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling
// RUN: cat %s | %cling 2>&1 | FileCheck %s
// Test handling and recovery from calling an unresolved symbol.

extern int foo(); // extern C++
void bar() { foo(); }
extern "C" int functionWithoutDefinition();

int i = 42;
i = functionWithoutDefinition();
// CHECK: IncrementalExecutor::executeFunction: symbol 'functionWithoutDefinition' unresolved while linking
i = foo();

extern "C" int printf(const char* fmt, ...);
printf("got i=%d\n", i); // CHECK: got i=42
int a = 12// CHECK: (int) 12

foo()
// CHECK: IncrementalExecutor::executeFunction: symbol '{{.*}}foo{{.*}}' unresolved
functionWithoutDefinition();
// CHECK: IncrementalExecutor::executeFunction: symbol 'functionWithoutDefinition' unresolved while linking

i = 13 //CHECK: (int) 13
.q
