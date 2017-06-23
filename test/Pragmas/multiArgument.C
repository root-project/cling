//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S -Xclang -verify 2>&1 | FileCheck %s

// FIXME: When printing can be properly unloaded don't force it here
"BEGIN"
// CHECK: (const char [6]) "BEGIN"

#pragma cling load("P0.h", "P1.h","P2.h")

ValueA
// CHECK-NEXT: (const char *) "ValueA"

ValueB
// CHECK-NEXT: (const char *) "ValueB"

ValueC
// CHECK-NEXT: (const char *) "ValueC"

.undo
.undo
.undo
.undo

// FIXME: When print Transactions are properly parenteted remove these
.undo
.undo
.undo

ValueA // expected-error {{use of undeclared identifier 'ValueA'}}

#pragma cling load "P0.h" "P1.h" "P2.h"
ValueA
// CHECK-NEXT: (const char *) "ValueA"

ValueB
// CHECK-NEXT: (const char *) "ValueB"

ValueC
// CHECK-NEXT: (const char *) "ValueC"

.undo
.undo
.undo
.undo

// FIXME: When print Transactions are properly parenteted remove these
.undo
.undo
.undo

ValueB // expected-error {{use of undeclared identifier 'ValueB'}}

#pragma cling(load, "P0.h", "P1.h", "P2.h")
ValueA
// CHECK-NEXT: (const char *) "ValueA"

ValueB
// CHECK-NEXT: (const char *) "ValueB"

ValueC
// CHECK-NEXT: (const char *) "ValueC"


#pragma cling load "P0.h P1.h P2.h"
ValueD
// CHECK-NEXT: (const char *) "ValueD"

.q
