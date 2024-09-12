//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s |  %cling 2>&1 | FileCheck %s

extern "C" int printf(const char*, ...);

struct A {
  int val;
  A(int v) : val(v) {
    printf("A(%d), this = %p\n", val, this);
  }
  ~A() {
    printf("~A(%d), this = %p\n", val, this);
  }
  int getVal() const { return val; }
};

const A a(1);
// CHECK: A(1), this = [[PTR:.+]]

a.val
// CHECK-NEXT: (const int) 1
a.getVal()
// CHECK-NEXT: (int) 1
a.val
// CHECK-NEXT: (const int) 1
a.getVal()
// CHECK-NEXT: (int) 1

// CHECK: ~A(1), this = [[PTR]]
// CHECK-NOT: ~A
