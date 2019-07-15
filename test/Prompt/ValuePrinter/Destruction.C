//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// Test valueDestruction

.rawInput

extern "C" int printf(const char*,...);

class A {
  int m_A[2] = {};
public:
  A() {}
  ~A() { printf("A::~A()\n"); }
};

class B : public A {
  int m_B[2];
public:
};

extern "C" {
  struct C {
    int m_C[4];
  };

  typedef struct {
    int C0, c1;
  } C2;
}

class D {
  int m_D[2];
public:
};

int gTest = 0;

class E {
  char mem = 0;
public:
  ~E() { gTest = 101; }
};

.rawInput

A()
//      CHECK: (A) @0x{{[0-9a-f]+}}
// CHECK-NEXT: A::~A()

B()
// CHECK-NEXT: (B) @0x{{[0-9a-f]+}}
// CHECK-NEXT: A::~A()

C()
// CHECK-NEXT: (C) @0x{{[0-9a-f]+}}

C2()
// CHECK-NEXT: (C2) @0x{{[0-9a-f]+}}

C2 c = {1, 2}
// CHECK-NEXT: (C2 &) @0x{{[0-9a-f]+}}

D()
// CHECK-NEXT: (D) @0x{{[0-9a-f]+}}

gTest
// CHECK-NEXT: 0

E()
// CHECK-NEXT: (E) @0x{{[0-9a-f]+}}

gTest
// CHECK-NEXT: 101

// Don't call destructor on printed lambda
[] {}
// CHECK-NEXT: () @0x{{[0-9a-f]+}}

// expected-no-diagnostics
.q
