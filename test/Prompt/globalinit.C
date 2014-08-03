//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling %s "globalinit(\"%s\")" | FileCheck %s

#include "cling/Interpreter/Interpreter.h"

void globalinit(const std::string& location) {
  gCling->loadFile(location + ".h", false); // CHECK: A::S()
  gCling->loadFile(location + "2.h", false); // CHECK: B::S()
}
// CHECK: B::~S()
// CHECK: A::~S()
