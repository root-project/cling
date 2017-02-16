//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S -Xclang -verify  | FileCheck %s
// Test externCUndo

.storeState "A"

extern "C" int printf(const char*,...);

.storeState "B"

extern "C" int printf(const char*,...);
extern "C" int printf(const char*,...);
extern "C" int printf(const char*,...);
extern "C" int printf(const char*,...);

extern "C" {
  int printf(const char*,...);
  int abs(int);
  double atof(const char *);
  void free(void* ptr);
}

.undo // extern "C" {}
.undo // printf
.undo // printf
.undo // printf
.undo // printf

.compareState "B"

printf("Unloaded alot\n");
// CHECK: Unloaded alot

.undo // printf()
.undo // printf

.compareState "A"

printf("FAIL\n"); // expected-error@2 {{use of undeclared identifier 'printf'}}

.q
