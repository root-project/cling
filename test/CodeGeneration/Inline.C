//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN:cat %s |  %cling -I %S -Xclang -verify 2>&1 | FileCheck -allow-empty %s
// Test testInline

// ROOT-8283

inline int localInline() { return 5; }
localInline();
localInline();
.undo
.undo
localInline();

extern "C" int printf(const char*, ...);

#include "Inline.h"
testInline(101)
// CHECK: 101
.undo

testInline(201)
// CHECK-NEXT: 201
testInline(202)
// CHECK-NEXT: 202
testInline(203)
// CHECK-NEXT: 203
.undo  // testInline()
.undo  // testInline()
.undo  // testInline()

testInline(301)
// CHECK-NEXT: 301
.undo  // testInline()

testInline(testInlineRet(301))
// CHECK-NEXT: 602
.undo  // testInline()

testInline(testInlineRet(401))
// CHECK-NEXT: 802
.undo  // testInline()

.undo  // #include "Inline.h"
testInline(501) // expected-error {{use of undeclared identifier 'testInline'}}

#include "Inline.h"
testInline(testInlineRet(606))
// CHECK-NEXT: 1212

.q
