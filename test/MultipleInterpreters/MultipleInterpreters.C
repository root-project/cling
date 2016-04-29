//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s

// Test to check the functionality of the multiple interpreters.
// Create a "child" interpreter and use gCling as its "parent".

#include "cling/Interpreter/Interpreter.h"

#include <cstdio>

const char* argV[1] = {"cling"};
cling::Interpreter ChildInterp(*gCling, 1, argV);

//Declare something in the parent interpreter
.rawInput 1
int foo(){ return 42; }
.rawInput 0
// OR
//gCling->declare("void foo(){ llvm::outs() << \"foo(void)\\n\"; }");

// Also declare something in the Child Interpreter
ChildInterp.declare("void foo(int i){ printf(\"foo(int) = %d\\n\", i); }");

//Then execute it from the child interpreter
ChildInterp.echo("foo()");
//CHECK:(int) 42

//Check if function overloading works
ChildInterp.execute("foo(1)");
//CHECK:foo(int) = 1
.q
