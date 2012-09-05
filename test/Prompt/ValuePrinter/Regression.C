// RUN: cat %s | %cling | FileCheck %s

// This file should be used as regression test for the value printing subsystem
// Reproducers of fixed bugs should be put here

// PR #93006
.rawInput 1
extern "C" int printf(const char* fmt, ...);
class A {
public:
  int Var;
  A(int arg) { Var = arg; }
  ~A() { printf("A d'tor\n"); }
};

const A& foo(const A& arg) { return arg; }
.rawInput 0

foo(A(12)).Var
// CHECK: (const int) 12
// CHECK: A d'tor
// End PR #93006

 // Savannah #96523
int *p = (int*)0x123;
p // CHECK: (int *) 0x123
const int *q = (int*)0x123;
q // CHECK: (const int *) 0x123
