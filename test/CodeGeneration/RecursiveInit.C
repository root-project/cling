//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling %s | FileCheck %s

extern "C" int printf(const char*,...);
#include "cling/Interpreter/Interpreter.h"

int DoRecurse() {
  gCling->process("int RecursiveInitVar1 = printf(\"Recursive init\\n\")");
  //CHECK: Recursive init
  //CHECK: (int) 15
  return 12;
}

const char* code =
  "DoRecurse();\n"
  "int RecursiveInitVar0 = 17;\n"
  "printf(\"%d\\n\", RecursiveInitVar0);";
// CHECK: 17

void RecursiveInit() {
  gCling->process(code);
}
