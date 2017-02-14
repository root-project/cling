//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %built_cling -Xclang -verify 2>&1 | FileCheck %s
// Make sure we are correctly parsing the arguments for CIFactory::createCI

#include "cling/Interpreter/InvocationOptions.h"

const char* argv[] = {
  "progname",
  "-",
  "-xobjective-c",
  "FileToExecuteA",
  "-isysroot",
  "APAth",
  "-nobuiltininc",
  "-v",
  "FileToExecuteB",
  "-L/Path/To/Libs",
  "-lTest"
};
const int argc = sizeof(argv)/sizeof(argv[0]);

cling::CompilerOptions COpts(argc, argv);

COpts.Language
// CHECK: (bool) true
COpts.SysRoot
// CHECK-NEXT: (bool) true
COpts.NoBuiltinInc
// CHECK-NEXT: (bool) true
COpts.NoCXXInc
// CHECK-NEXT: (bool) false

// library caller options: arguments passed as is
COpts.Remaining
// CHECK-NEXT: {{.*}} { "progname", "-", "-xobjective-c", "FileToExecuteA", "-isysroot", "APAth", "-nobuiltininc", "-v", "FileToExecuteB", "-L/Path/To/Libs", "-lTest" }

argv[6] = "-nostdinc++";
cling::InvocationOptions IOpts(argc, argv);
IOpts.Inputs
// CHECK-NEXT: {{.*}} { "-", "FileToExecuteA", "FileToExecuteB" }

IOpts.LibSearchPath
// CHECK-NEXT: {{.*}} { "/Path/To/Libs" }

IOpts.LibsToLoad
// CHECK-NEXT: {{.*}} { "Test" }

IOpts.CompilerOpts.Language
// CHECK-NEXT: (bool) true
IOpts.CompilerOpts.SysRoot
// CHECK-NEXT: (bool) true
IOpts.CompilerOpts.NoBuiltinInc
// CHECK-NEXT: (bool) false
IOpts.CompilerOpts.NoCXXInc
// CHECK-NEXT: (bool) true

// user options from main: filtered by cling (no '-')
IOpts.CompilerOpts.Remaining

// Windows translates -nostdinc++ to -nostdinc++. Ignore that fact for the test.
// CHECK-NEXT: {{.*}} { "progname", "-xobjective-c", "FileToExecuteA", "-isysroot", "APAth", {{.*}}, "-v", "FileToExecuteB" }

// expected-no-diagnostics
.q
