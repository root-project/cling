//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p -Xclang -verify 2>&1 | FileCheck %s

// Test the removal of decls which are stored in vector of redeclarables
.rawInput 1
extern int __my_i;
template<typename T> T TemplatedF(T t);
template<> double TemplatedF(double); // forward declare TemplatedF
int OverloadedF(int i);
float OverloadedF(float f){ return f + 100.111f;}
double OverloadedF(double d){ return d + 10.11f; };
namespace test { int y = 0; }
.rawInput 0

.storeState "testSubsequentDecls"
#include "SubsequentDecls.h"
.compareState "testSubsequentDecls"
// CHECK-NOT: Differences

TemplatedF((int)2)
// CHECK: IncrementalExecutor::executeFunction: symbol '{{.*}}' unresolved while linking [cling interface function]!
// CHECK: You are probably missing the definition of int {{.*}}TemplatedF<int>(int)
// CHECK: Maybe you need to load the corresponding shared library?

template<> int TemplatedF(int i) { return i + 100; }
int OverloadedF(int i) { return i + 100;}

int __my_i = 10
// CHECK: (int) 10
OverloadedF(__my_i)
// CHECK: (int) 110
TemplatedF(__my_i)
// CHECK: (int) 110

TemplatedF((double)3.14)
// CHECK: IncrementalExecutor::executeFunction: symbol '{{.*}}' unresolved while linking [cling interface function]!
// CHECK: You are probably missing the definition of double {{.*}}TemplatedF<double>(double)
// CHECK: Maybe you need to load the corresponding shared library?

// ROOT-7295
#include <vector>
std::vector<float> a; // expected-note {{previous definition is here}}
std::vector<float> a; // expected-error {{redefinition of 'a'}}

.q
