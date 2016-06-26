//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p 2>&1 | FileCheck %s
// Test that we re-read file, e.g. that we uncache symbols and file content.

// ROOT-7858: forget symbols
.L macro1.h
macro() // CHECK: version 1
//CHECK: (int) 1

.U macro1.h
.L macro2.C
macro() // CHECK: 2.version 2
//CHECK: (int) 2
