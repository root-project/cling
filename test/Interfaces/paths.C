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

// Paths that are only spelled differently are not added again.
gCling->AddIncludePath(TEST_PATH "Paths//D/");
gCling->AddIncludePaths(TEST_PATH "Paths/./A:" TEST_PATH "Paths/E:"
                        TEST_PATH "Paths/E/");
#ifdef _WIN32
gCling->AddIncludePath(TEST_PATH "PATHS\\D");
#endif
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <string>
llvm::SmallVector<std::string, 32> IncPaths;
gCling->GetIncludePaths(IncPaths, false, false);
(int)std::count_if(IncPaths.begin(), IncPaths.end(), [](const std::string& P) {
  return P.find("Paths") != std::string::npos ||
         P.find("PATHS") != std::string::npos;
})
// CHECK: (int) 5

// expected-no-diagnostics
.q
