//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -DTEST_PATH="\"%/p/\"" -Xclang -verify 2>&1 | FileCheck %s

#include "cling/Interpreter/Interpreter.h"

gCling->AddIncludePaths(TEST_PATH "Paths/A:" TEST_PATH "Paths/B:"
                        TEST_PATH "Paths/C");
#include "A.h"
#include "B.h"
#include "C.h"

gCling->AddIncludePath(TEST_PATH "Paths/D");
#include "D.h"

TestA
// CHECK: (const char *) "TestA"
TestB
// CHECK: (const char *) "TestB"
TestC
// CHECK: (const char *) "TestC"
TestD
// CHECK: (const char *) "TestD"

// expected-no-diagnostics
.q
