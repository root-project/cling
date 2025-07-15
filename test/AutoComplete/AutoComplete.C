//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s

// Test to check autocomplete functionality (adapted from CppInterOp)

#include "cling/Interpreter/Interpreter.h"

gCling->process("int foo = 12;");
std::vector<std::string> cc;
size_t cursor = 1;
gCling->codeComplete("f", cursor, cc);

// We check only for 'float' and 'foo', because they
// must be present in the result. Other hints may appear
// there, depending on the implementation, but these two
// are required to say that the test is working.
size_t cnt = 0;
for (auto& r : cc) {
  if (r == "float" || r == "[#int#]foo") cnt++;
}

cnt
// CHECK: (unsigned long) 2
.q
