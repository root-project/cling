// RUN: cat %s | %cling
// RUN: cat %s | %cling 2>&1 | FileCheck %s
// Test handling and recovery from calling an unresolved symbol.

.rawInput
int foo(); // extern C++
.rawInput
extern "C" int functionWithoutDefinition();

int i = 42;
i = functionWithoutDefinition();
// CHECK: ExecutionContext::executeFunction: symbol 'functionWithoutDefinition' unresolved
i = foo();
// CHECK: ExecutionContext::executeFunction: symbol '{{.*}}foo{{.*}}' unresolved

extern "C" int printf(const char* fmt, ...);
printf("got i=%d\n", i); // CHECK: got i=42
int a = 12// CHECK: (int) 12

foo()
// CHECK: ExecutionContext::executeFunction: symbol '{{.*}}foo{{.*}}' unresolved
functionWithoutDefinition();
// CHECK: ExecutionContext::executeFunction: symbol 'functionWithoutDefinition' unresolved

i = 13 //CHECK: (int) 13
