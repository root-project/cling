//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling --noruntime -Xclang -verify  | FileCheck %s
// Test unloadTypeInfo

extern "C" int printf(const char*, ...);
int A, B;

#include <typeinfo>
printf("Types match: %d\n", typeid(A) == typeid(B));
// CHECK: Types match: 1

.undo  // undo print
.undo  // undo #include <typeinfo>

#include <typeinfo>
printf("Types match: %d\n", typeid(A) == typeid(B));
// CHECK: Types match: 1

// expected-no-diagnostics
.q
