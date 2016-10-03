//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling %s -I%p | FileCheck %s

#include "cling/Interpreter/Interpreter.h"

void globalinit() {
  gCling->loadFile("globalinit.C.h", false); // CHECK: A::S()
  gCling->loadFile("globalinit.C2.h", false); // CHECK: B::S()
}
// CHECK: B::~S()
// CHECK: A::~S()
