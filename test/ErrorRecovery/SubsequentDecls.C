// RUN: cat %s | %cling -Xclang -verify -I%p | FileCheck %s
// Test the removal of decls which are stored in vector of redeclarables 

.rawInput 1
extern int __my_i;
template<typename T> T TemplatedF(T t);
int OverloadedF(int i);
float OverloadedF(float f){ return f + 100.111f;}
double OverloadedF(double d){ return d + 10.11f; };
namespace test { int y = 0; }
.rawInput 0

#include "SubsequentDecls.h"

.rawInput 1
template<> int TemplatedF(int i) { return i + 100; }
int OverloadedF(int i) { return i + 100;}
.rawInput 0

int __my_i = 10
// CHECK: (int) 10
OverloadedF(__my_i)
// CHECK: (int const) 110
TemplatedF(__my_i)
// CHECK: (int const) 110

.q
