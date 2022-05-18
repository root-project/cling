//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

// Test that the interpreter properly generates implicitly defined destructors
// overriding the virtual destructor in the base class.

// expected-no-diagnostics

.rawInput
extern "C" int printf(const char*,...);

class A {
public:
  virtual ~A() {
    printf("A::~A()\n");
  }
};

class B : public A {
public:
  B() {
    printf("B::B()\n");
  }
};
.rawInput

// CHECK-NOT: error
// CHECK-NOT: Symbols not found

// CHECK: B::B()
A *b = new B;
// CHECK: A::~A()
delete b;
