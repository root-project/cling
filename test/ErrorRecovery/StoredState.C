//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -v -Xclang -verify 2>&1 | FileCheck %s

int i = 0;
.storeState "A"

.undo
// CHECK: Unloading Transaction forced state 'A' to be destroyed

// expected-no-diagnostics

.q
