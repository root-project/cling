//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

// Test to check functions registered via atexit are intercepted, and __dso_handle
// is properly overridden in for child interpreters.
#include <cstdlib>
#include "cling/Interpreter/Interpreter.h"


static void atexit_1() {
  printf("atexit_1\n");
}
static void atexit_2() {
  printf("atexit_2\n");
}
static void atexit_3() {
  printf("atexit_3\n");
}

atexit(atexit_1);
.undo
// Undoing the registration should call the function
// CHECK: atexit_1

at_quick_exit(atexit_2);
.undo
// Make sure at_quick_exit is resolved correctly (mangling issues on gcc < 5)
// CHECK-NEXT: atexit_2

atexit(atexit_3);

cling::Interpreter * gChild = 0;
{
  const char* kArgV[1] = {"cling"};
  cling::Interpreter ChildInterp(*gCling, 1, kArgV);
  gChild = &ChildInterp;
  ChildInterp.declare("static void atexit_c() { printf(\"atexit_c %d\\n\", gChild==__dso_handle); }");
  ChildInterp.execute("atexit(atexit_c);");
}
// ChildInterp
// CHECK-NEXT: atexit_c 1

static void atexit_f() {
  printf("atexit_f %s\n", gCling==__dso_handle ? "true" : "false");
}
at_quick_exit(atexit_f);

// expected-no-diagnostics
.q

// Reversed registration order
// CHECK-NEXT: atexit_f true
// CHECK-NEXT: atexit_3
