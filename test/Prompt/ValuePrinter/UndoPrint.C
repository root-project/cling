//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// Test undoPrinter

.stats undo
//      CHECK: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>

const char *message = "winkey";

message
// CHECK-NEXT: (const char *) "winkey"

.undo

// Make sure we can still print
message
// CHECK-NEXT: (const char *) "winkey"

.undo
.stats undo
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>

message
// CHECK-NEXT: (const char *) "winkey"

.undo // print message
.undo // decalre message
.stats undo
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>

#include "cling/Interpreter/Interpreter.h"

gCling->echo("1;");
// CHECK-NEXT: (int) 1

.stats undo
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `      <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `      <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `      <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>

.undo
.stats undo
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>

gCling->echo("1;");
// CHECK-NEXT: (int) 1

// expected-no-diagnostics
.q
