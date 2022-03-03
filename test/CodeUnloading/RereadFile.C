//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p 2>&1 | FileCheck %s
// Test that we re-read file, e.g. that we uncache symbols and file content.

extern "C" int printf(const char*,...);

// work around ROOT-8240
42 // CHECK: (int) 42

// ROOT-7858: forget symbols
.L macro1.h
macro() // CHECK: version 1
//CHECK: (int) 1

.U macro1.h
.L macro2.h
macro() // CHECK: 2.version 2
//CHECK: (int) 2

.x unnamedns.h
//CHECK: 13
.x unnamedns.h
//CHECK-NEXT: 13

.x templatedfunc.h
//CHECK: 4
.x templatedfunc.h
//CHECK-NEXT: 4
