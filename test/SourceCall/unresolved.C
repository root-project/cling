// RUN: cat %s | %cling
// RUN: cat %s | %cling 2>&1 | FileCheck %s
// Test handling and recovery from calling an unresolved symbol.

.rawInput
int foo(); // extern C++
void bar() { foo(); }
.rawInput
extern "C" int functionWithoutDefinition();

int i = 42;
i = functionWithoutDefinition();
// CHECK: ExecutionContext: use of undefined symbol 'functionWithoutDefinition'!
// CHECK: ExecutionContext::executeFunction: symbol 'functionWithoutDefinition' unresolved while linking function 
i = foo();
// CHECK: ExecutionContext: use of undefined symbol '{{.*}}foo{{.*}}'!
// CHECK: ExecutionContext::executeFunction: symbol '{{.*}}foo{{.*}}' unresolved while linking function 

extern "C" int printf(const char* fmt, ...);
printf("got i=%d\n", i); // CHECK: got i=42
int a = 12// CHECK: (int) 12

foo()
// CHECK: ExecutionContext::executeFunction: symbol '{{.*}}foo{{.*}}' unresolved
functionWithoutDefinition();
// CHECK: ExecutionContext::executeFunction: symbol 'functionWithoutDefinition' unresolved while linking function 

bar();
// CHECK: ExecutionContext: use of undefined symbol '{{.*}}foo{{.*}}'!
// CHECK: ExecutionContext::executeFunction: symbol '{{.*}}foo{{.*}}' unresolved while linking function 
bar();
// CHECK: ExecutionContext: calling unresolved symbol, see previous error message!

i = 13 //CHECK: (int) 13
