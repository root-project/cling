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
#include "llvm/Support/raw_ostream.h"


//Declare something in the parent interpreter
.rawInput 1
int foo(){ return 42; }
.rawInput 0
// OR
//gCling->declare("void foo(){ llvm::outs() << \"foo(void)\\n\"; }");

const char* argV[1] = {"cling"};
using namespace cling;
Interpreter::CompilationResult compRes = Interpreter::kSuccess;
Interpreter ChildInterp(*gCling, 1, argV);
MetaProcessor ChildMP(ChildInterp, llvm::errs());

// Also declare something in the Child Interpreter
// Then execute it from the child interpreter
// And check if function overloading works.
// All needs to happen in one statement, or else the contract
// that the parent is not modified during the child's lifetime
// is violated.
ChildMP.process(".rawInput 1\n"
                "void foo(int i){ printf(\"foo(int) = %d\\n\", i); }\n"
                ".rawInput 0\n"
                "foo()\n" //CHECK: (int) 42
                "foo(1)",
                compRes, nullptr
); //CHECK: foo(int) = 1
compRes // CHECK: cling::Interpreter::CompilationResult::kSuccess
.q
