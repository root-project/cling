//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p 2>&1 | FileCheck %s

// The main issue is that expected - error is not propagated to the source file and
// the expected diagnostics get misplaced.

.x CannotDotX.h()
// expected-warning{{'CannotDotX' missing falling back to .L}}

// Here we cannot revert MyClass from CannotDotX.h
.L CannotDotX.h
MyClass m;
// CHECK: MyClass ctor called
.L CannotDotX.h
// CHECK: MyClass dtor called
.q
