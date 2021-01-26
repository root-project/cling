//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s
extern "C" int printf(const char*,...);

// When interpreting code, raised exceptions can be catched by the call site.

#include "cling/Interpreter/Interpreter.h"

try {
  gCling->process("throw 1;");
} catch (...) {
  // CHECK: Caught exception from throw statement
  printf("Caught exception from throw statement\n");
}

struct ThrowInConstructor {
  ThrowInConstructor() { throw 1; }
};
try {
  gCling->process("ThrowInConstructor t;");
} catch (...) {
  // CHECK: Caught exception from constructor
  printf("Caught exception from constructor\n");
}

.q
