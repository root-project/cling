//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling --noruntime -Xclang -verify 2>&1 | FileCheck --allow-empty %s
// Test unloadForwardReturn

.storeState "A"

.rawInput
struct TEST* funcReturnA() { return 0; }
struct TEST* funcReturnB() { return 0; }
struct TEST* funcReturnC() { return 0; }
.rawInput

.undo
.undo
.undo
.compareState "A"
// CHECK-NOT: Differences

.rawInput
struct TEST* funcReturnA() { return 0; }
struct TEST { int a, b, c; };
.storeState "B"
struct TEST* funcReturnB() { return 0; }
struct TEST* funcReturnC() { return 0; }
.rawInput

.undo
.undo
.compareState "B"
// CHECK-NOT: Differences

TEST t = { 30, 40, 50 };
.undo
.undo
.undo

.compareState "A"
// CHECK-NOT: Differences

// expected-no-diagnostics
.q
