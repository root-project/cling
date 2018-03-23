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
// CHECK: (unsigned int) 1
COpts.SysRoot
// CHECK-NEXT: (unsigned int) 1
COpts.NoBuiltinInc
// CHECK-NEXT: (unsigned int) 1
COpts.NoCXXInc
// CHECK-NEXT: (unsigned int) 0
COpts.CUDA
// CHECK-NEXT: (unsigned int) 0

COpts.DefaultLanguage()
// CHECK-NEXT: false

// library caller options: arguments passed as is
COpts.Remaining
// CHECK-NEXT: {{.*}} { "progname", "-", "-xobjective-c", "FileToExecuteA", "-isysroot", "APAth", "-nobuiltininc", "-v", "FileToExecuteB", "-L/Path/To/Libs", "-lTest" }

argv[2] = "-xcuda";
argv[6] = "-nostdinc++";

cling::InvocationOptions IOpts(argc, argv);
IOpts.Inputs
// CHECK-NEXT: {{.*}} { "-", "FileToExecuteA", "FileToExecuteB" }

IOpts.LibSearchPath
// CHECK-NEXT: {{.*}} { "/Path/To/Libs" }

IOpts.LibsToLoad
// CHECK-NEXT: {{.*}} { "Test" }

IOpts.CompilerOpts.Language
// CHECK-NEXT: (unsigned int) 1
IOpts.CompilerOpts.SysRoot
// CHECK-NEXT: (unsigned int) 1
IOpts.CompilerOpts.NoBuiltinInc
// CHECK-NEXT: (unsigned int) 0
IOpts.CompilerOpts.NoCXXInc
// CHECK-NEXT: (unsigned int) 1
IOpts.CompilerOpts.CUDA
// CHECK-NEXT: (unsigned int) 1

// if the language is cuda, it should set automatically the c++ standard
IOpts.CompilerOpts.DefaultLanguage()
// CHECK-NEXT: true

// user options from main: filtered by cling (no '-')
IOpts.CompilerOpts.Remaining

// Windows translates -nostdinc++ to -nostdinc++. Ignore that fact for the test.
// CHECK-NEXT: {{.*}} { "progname", "-xcuda", "FileToExecuteA", "-isysroot", "APAth", {{.*}}, "-v", "FileToExecuteB" }

// expected-no-diagnostics
.q
