//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s
extern "C" int printf(const char*,...);
extern "C" void exit(int);

int i;
struct S{int i;} s;
i = 42;
printf("i=%d\n",i); // CHECK: i=42
if (i != 42) exit(1);
.q
