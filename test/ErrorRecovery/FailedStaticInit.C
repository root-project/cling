//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// A failed static initializer rolls the transaction back and is reported.

#include "cling/Interpreter/Interpreter.h"

gCling->declare("extern \"C\" int unresolvedInitFn(); int unresolvedInit = unresolvedInitFn();")
// CHECK: (cling::Interpreter::CompilationResult) (cling::Interpreter::kFailure) : ({{(unsigned )?}}int) 1

unresolvedInit // expected-error {{use of undeclared identifier 'unresolvedInit'}}

// The name and the symbol are free again.
int unresolvedInit = 1;
unresolvedInit
// CHECK: (int) 1

gCling->process("extern \"C\" int unresolvedInitFn2(); int unresolvedInit2 = unresolvedInitFn2();")
// CHECK: (cling::Interpreter::CompilationResult) (cling::Interpreter::kFailure) : ({{(unsigned )?}}int) 1

int initSource() { return 7; }
int initAfterFailure = initSource();
initAfterFailure
// CHECK: (int) 7
.q
