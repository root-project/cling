// RUN: cat %s | %cling 2>&1 | FileCheck %s

// Test the ability of unloading the last transaction. Here as a matter of fact
// we unload the wrapper as well and TODO: decrement the unique wrapper counter.
extern "C" int printf(const char* fmt, ...);
printf("Force printf codegeneration. Otherwise CG will defer it and .storeState will be unhappy.\n");
//CHECK: Force printf codegeneration. Otherwise CG will defer it and .storeState will be unhappy.
.storeState "preUnload"
int f = 0;
.U
.rawInput 1
int f() {
  printf("Now f is a function\n");
  return 0;
} int a = f();
.U
.rawInput 0
//CHECK: Now f is a function
.compareState "preUnload"
//CHECK-NOT: Differences
double f = 3.14
//CHECK: (double) 3.14
