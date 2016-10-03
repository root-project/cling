//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling %s -DTEST_PATH="\"%/p/\"" -Xclang -verify 2>&1 | FileCheck %s
// RUN: %cling %s -I%p -Xclang -verify 2>&1 | FileCheck %s

#include "cling/Interpreter/Interpreter.h"

#ifndef TEST_PATH
  #define TEST_PATH ""
#endif

void globalinit(const std::string Path = TEST_PATH) {
  gCling->loadFile(Path + "globalinit.C.h", false); // CHECK: A::S()
  gCling->loadFile(Path + "globalinit.C2.h", false); // CHECK: B::S()
}
// CHECK: B::~S()
// CHECK: A::~S()

// expected-no-diagnostics
