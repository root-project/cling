//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S --noruntime -Xclang -verify 2>&1 | FileCheck %s
// Test unloadNew

extern "C" int printf(const char*,...);

#include <new>
.undo

#include "OpNew.h"
Test *t = new Test;
// CHECK: Test::operator new

delete t;
// CHECK: Test::operator delete

.undo // delete t;
.undo // new Test;
.undo // #include "OpNew.h"

#include <new>
.undo //#include <new>

// Now we are at line 13
#include <new>

.rawInput
// CHECK: Using raw input
static unsigned long allocFact(unsigned N) {
  unsigned *test = new unsigned[N];
  for (int i = 0; i < N; ++i)
    test[i] = i+1;

  unsigned long fact = 1;
  for (int i = 0; i < N; ++i)
    fact *= test[i];

  delete [] test;
  return fact;
}

static unsigned long factorial(unsigned long n) {
  return (n < 2) ? 1 : (factorial(n - 1) * n);
}
.rawInput
// CHECK: Not using raw input

printf("%lu == %lu\n", allocFact(12), factorial(12));
// CHECK: 479001600 == 479001600

// expected-no-diagnostics
.q
