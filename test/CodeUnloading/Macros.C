//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S -Xclang -verify 2>&1 | FileCheck %s
// Tests undoing of macro definitions

// Invoke the printer to get it in the undo queue early
"TEST"
// CHECK: (const char [5]) "TEST"

// Make sure one Transactin can handle redefinitions
#include "Macros.h"
// expected-warning@Macros.h:3 {{'TEST' macro redefined}}
// expected-note@Macros.h:2 {{previous definition is here}}
// expected-warning@Macros.h:4 {{'TEST' macro redefined}}
// expected-note@Macros.h:3 {{previous definition is here}}
// expected-warning@Macros.h:5 {{'TEST' macro redefined}}
// expected-note@Macros.h:4 {{previous definition is here}}
// expected-warning@Macros.h:6 {{'TEST' macro redefined}}
// expected-note@Macros.h:5 {{previous definition is here}}

TEST
// CHECK: (const char [7]) "TEST 4"

.undo //print
.undo //include
.undo // FIXME: REMOVE once print unloading is merged

TEST // expected-error@2 {{use of undeclared identifier 'TEST'}}

#define TEST "DEFINED"
#undef TEST
.undo
TEST
// CHECK: (const char [8]) "DEFINED"
.undo // print
.undo // define
.undo // FIXME: REMOVE once print unloading is merged

TEST // expected-error@2 {{use of undeclared identifier 'TEST'}}

// Make sure one Transactin can handle undef, redef
#define TESTB
#include "Macros.h"
// expected-warning@Macros.h:19 {{'TEST' macro redefined}}
// expected-note@Macros.h:18 {{previous definition is here}}

TEST // CHECK: (const char [7]) "TEST G"
.q
