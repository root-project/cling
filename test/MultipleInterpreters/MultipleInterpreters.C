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
#include "cling/MetaProcessor/MetaProcessor.h"
// #include "cling/Utils/Output.h"


//Declare something in the parent interpreter
int foo(){ return 42; }

// OR
//gCling->declare("void foo(){ cling::outs() << \"foo(void)\\n\"; }");

const char* argV[1] = {"cling"};
// Declare something in the child interpreter, then execute it from the child
// interpreter, and check if function overload resolution works.
// All needs to happen in one parent statement, or else the contract
// that the parent is not modified during the child's lifetime
// is violated.
{
  cling::Interpreter ChildInterp(*gCling, 1, argV);
  ChildInterp.declare("void foo(int i){ printf(\"foo(int) = %d\\n\", i); }\n");
  ChildInterp.echo("foo()"); //CHECK: (int) 42
  ChildInterp.echo("foo(1)"); //CHECK: foo(int) = 1
}
.q
