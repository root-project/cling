//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s

.storeState "preUnload"
.rawInput 1
int g(); int f(int i) { if (i != 1) return g(); return 0; } int g() { return f(1); } int x = f(0);
.rawInput 0
.undo
.compareState "preUnload"
//CHECK-NOT: Differences
float f = 1.1
//CHECK: (float) 1.1
int g = 42
//CHECK: (int) 42
.q
