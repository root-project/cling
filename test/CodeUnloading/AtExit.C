//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// FIXME: cat %s | %cling -fsyntax-only -Xclang -verify 2>&1

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

// Test reverse ordering in a single transaction.
static void atexitA() { printf("atexitA\n"); }
static void atexitB() { printf("atexitB\n"); }
static void atexitC() { printf("atexitC\n"); }
{
  std::atexit(atexitA);
  std::atexit(atexitB);
  std::atexit(atexitC);
}
.undo
// CHECK-NEXT: atexitC
// CHECK-NEXT: atexitB
// CHECK-NEXT: atexitA

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
// CHECK: atexit_c 1

static void atexit_f() {
  printf("atexit_f %s\n", gCling==__dso_handle ? "true" : "false");
}
at_quick_exit(atexit_f);

void atExit0 () {
  printf("atExit0\n");
}
void atExit1 () {
  printf("atExit1\n");
}
void atExit2 () {
  printf("atExit2\n");
}
void atExitA () {
  printf("atExitA\n");
  std::atexit(&atExit0);
}
void atExitB () {
  printf("atExitB\n");
  std::atexit(&atExit1);
  std::atexit(&atExit2);
}
// Recursion in a Transaction
{
  std::atexit(&atExitA);
  std::atexit(&atExitB);
}
.undo
// CHECK-NEXT: atExitB
// CHECK-NEXT: atExit2
// CHECK-NEXT: atExit1
// CHECK-NEXT: atExitA
// CHECK-NEXT: atExit0

// Recusion at shutdown
struct ShutdownRecursion {
  static void DtorAtExit0() { printf("ShutdownRecursion0\n"); }
  static void DtorAtExit1() { printf("ShutdownRecursion1\n"); }
  static void DtorAtExit2() { printf("ShutdownRecursion2\n"); }
  ~ShutdownRecursion() {
    printf("~ShutdownRecursion\n");
    atexit(&DtorAtExit0);
    atexit(&DtorAtExit1);
    atexit(&DtorAtExit2);
  }
} Out;

// expected-no-diagnostics
.q

// Reversed registration order

// CHECK-NEXT: ~ShutdownRecursion
// CHECK-NEXT: ShutdownRecursion2
// CHECK-NEXT: ShutdownRecursion1
// CHECK-NEXT: ShutdownRecursion0

// CHECK-NEXT: atexit_f true
// CHECK-NEXT: atexit_3
