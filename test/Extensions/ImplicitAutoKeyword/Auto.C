//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling --enable-implicit-auto-keyword | FileCheck %s
// XFAIL: *

// ROOT-5324
b = 1
//CHECK: (int) 1
b += 1
//CHECK: (int) 2
b = 0
//CHECK: (int) 0
a = b
//CHECK: (int) 0
int c = a
//CHECK: (int) 0
// end ROOT-5324

auto explicitAuto = "test";
explicitAuto
//CHECK: (const char*) "test"

.q
