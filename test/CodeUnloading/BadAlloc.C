//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling --noruntime -Xclang -verify 2>&1 | FileCheck %s
// Test unloadStdBadAlloc

extern "C" int printf(const char*,...);

#include <new>
try {
  throw std::bad_alloc();
} catch (const std::bad_alloc& e) {
  printf("CAUGHT 0\n");
  // CHECK: CAUGHT 0
}
.undo
.undo

#include <new>
try {
  throw std::bad_alloc();
} catch (const std::bad_alloc& e) {
  printf("CAUGHT 1\n");
  // CHECK: CAUGHT 1
}
.undo
.undo

#include <new>
try {
  throw std::bad_alloc();
} catch (const std::exception& e) {
  printf("CAUGHT 2\n");
  // CHECK: CAUGHT 2
}

printf("OUT\n");
// CHECK: OUT

// expected-no-diagnostics
.q
