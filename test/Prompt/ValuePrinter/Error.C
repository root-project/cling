//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"
#include <string>
#include <stdexcept>
#include <stdio.h>

class Thrower {
  int m_Private = 0;
public:
  Thrower(int I = 1) : m_Private(I) {
    if (I) throw std::runtime_error("Thrower");
  }
  ~Thrower() { printf("~Thrower-%d\n", m_Private); }
};

void barrier() {
  static int N = 0;
  printf("%d -------------\n", N++);
}

namespace cling {
  std::string printValue(const Thrower* T) {
    throw std::runtime_error("cling::printValue");
    return "";
  }
}

barrier();
//      CHECK: 0 -------------


// Un-named, so it's not a module static which would trigger std::terminate.
Thrower()
// CHECK-NEXT: >>> Caught a std::exception: 'Thrower'.
//  CHECK-NOT: ~Thrower-1

barrier();
// CHECK-NEXT: 1 -------------

Thrower& fstatic() {
  static Thrower sThrower;
  return sThrower;
}
fstatic()
// CHECK-NEXT: >>> Caught a std::exception: 'Thrower'.
//  CHECK-NOT: ~Thrower-1

barrier();
// CHECK-NEXT: 2 -------------

// Must be -new-, throwing from a constructor of a static calls std::terminate!
new Thrower
// CHECK-NEXT: >>> Caught a std::exception: 'Thrower'.
//  CHECK-NOT: ~Thrower-1

barrier();
// CHECK-NEXT: 3 -------------

cling::Value V;
gCling->evaluate("Thrower T1(1)", V);
// CHECK-NEXT: >>> Caught a std::exception: 'Thrower'.
V = cling::Value();
//  CHECK-NOT: ~Thrower-1

barrier();
// CHECK-NEXT: 4 -------------

gCling->evaluate("Thrower()", V);
// CHECK-NEXT: >>> Caught a std::exception: 'Thrower'.
V = cling::Value();
//  CHECK-NOT: ~Thrower-1

barrier();
// CHECK-NEXT: 5 -------------


gCling->evaluate("Thrower T1(1)", V);
// CHECK-NEXT: >>> Caught a std::exception: 'Thrower'.
V = cling::Value();
//  CHECK-NOT: ~Thrower-1

barrier();
// CHECK-NEXT: 6 -------------


// Check throwing from cling::printValue doesn't crash.
Thrower T0(0)
// CHECK-NEXT: >>> Caught a std::exception: 'cling::printValue'.

barrier();
// CHECK-NEXT: 7 -------------

gCling->echo("Thrower T0a(0)");
// CHECK-NEXT: ~Thrower-0
// CHECK-NEXT: >>> Caught a std::exception: 'cling::printValue'.


barrier();
// CHECK-NEXT: 8 -------------

gCling->echo("Thrower(0)");
// CHECK-NEXT: ~Thrower-0
// CHECK-NEXT: >>> Caught a std::exception: 'cling::printValue'.

barrier();
// CHECK-NEXT: 9 -------------


// T0 is a valid object and destruction should occur when out of scope.
//  CHECK-NOT: ~Thrower-1
// CHECK-NEXT: ~Thrower-0
//  CHECK-NOT: ~Thrower-1

// expected-no-diagnostics
.q
