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
.storeState "B"
.storeState "C"

.undo
// CHECK: Unloading Transaction forced state 'A' to be destroyed
// CHECK-NEXT: Unloading Transaction forced state 'B' to be destroyed
// CHECK-NEXT: Unloading Transaction forced state 'C' to be destroyed

.compareState "D"
.compareState "E"
.compareState "F"

// CHECK-NEXT: The store point name D does not exist.Unbalanced store / compare
// CHECK-NEXT: The store point name E does not exist.Unbalanced store / compare
// CHECK-NEXT: The store point name F does not exist.Unbalanced store / compare

// expected-no-diagnostics

.q
