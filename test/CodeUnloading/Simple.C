// RUN: cat %s | %cling | FileCheck %s

// Test the ability of unloading the last transaction. Here as a matter of fact
// we unload the wrapper as well and TODO: decrement the unique wrapper counter.
int f = 0;
.U
.rawInput 1
extern "C" int printf(const char* fmt, ...);
int f() {
  printf("Now f is a function\n");
  return 0;
} int a = f();
.U
.rawInput 0
//CHECK: Now f is a function
double f = 3.14
//CHECK: (double) 3.14
