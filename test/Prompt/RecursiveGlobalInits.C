//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s

// Checks for infinite recursion when we combine nested calls of process line
// with global initializers.

#include "cling/Interpreter/Interpreter.h"

class MyClass { public:  MyClass(){ gCling->process("gCling->getVersion()");} };

MyClass *My = new MyClass(); // CHECK: (const char *) "{{.*}}"

.q
