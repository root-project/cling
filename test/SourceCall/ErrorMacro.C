//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

.rawInput 1
int example() { } // expected-error {{control reaches end of non-void function}}
.rawInput 0
// Make FileCheck happy with having at least one positive rule:
int a = 5
// CHECK: (int) 5
.q
