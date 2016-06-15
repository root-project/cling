//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S -Xclang -verify 2>&1 | FileCheck %s
// Test unloadInitializerList

#include "InitializerList.h"
TestIList<int,10> t = {0,1,2,3,4,5,6,7,8,9};
t.sum()
// CHECK: (int) 45
.undo
.undo
.undo

#include "InitializerList.h"
TestIList<int,15> t = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
t.sum()
// CHECK-NEXT: (int) 120
.undo
.undo
.undo

#include "InitializerList.h"
TestIList<int,20> ta = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
ta.sum()
// CHECK-NEXT: (int) 210

TestIList<int,4> tb = {1,2,3,4};
tb.sum()
// CHECK-NEXT: (int) 10

TestIList<float,4> tf = {10,20,30,40};
tf.sum()
// CHECK-NEXT: (float) 100.00000f

// Still busted because of inlining problems
// #include <vector>
// std::vector<int> v = { 0, 1, 2, 3, 4, 5 };
// .undo
// .undo

// #include <vector>
// std::vector<int> v1 = { 0, 1, 2, 3, 4, 5 };
// v1
// // 0CHECK: (std::vector<int> &) { 0, 1, 2, 3, 4, 5 }

// expected-no-diagnostics
.q
