//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

extern "C" int printf(const char*, ...);
const char* TEST = "1234";
printf("val: %s\n", TEST);
//      CHECK: val: 1234
// .stats undo

TEST = "4567";
printf("val: %s\n", TEST);
// CHECK-NEXT: val: 4567

TEST = "8910";
printf("val: %s\n", TEST);
// CHECK-NEXT: val: 8910

.undo
.undo
printf("val: %s\n", TEST);
// CHECK-NEXT: val: 4567

.undo
.undo
.undo
// .stats undo

printf("val: %s\n", TEST);
// CHECK-NEXT: val: 1234

// expected-no-diagnostics
.q
