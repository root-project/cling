//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// Test Check the ability to undo past runtime printing.

// FIXME:
// Unloading past first print Transaction can fail due to decl unloading.
// Currently this test only validates that printing Transactions are properly
// compressed/parented into one atomic undo-able Transaction.

.stats undo
//      CHECK: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>

struct Trigger {};
Trigger T0
// CHECK-NEXT: (Trigger &) @0x{{[0-9a-f]+}}

.stats undo
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>


Trigger T1
// CHECK-NEXT: (Trigger &) @0x{{[0-9a-f]+}}
.stats undo
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>
// CHECK-NEXT: `   <cling::Transaction* 0x{{[0-9a-f]+}} isEmpty=0 isCommitted=1>


// expected-no-diagnostics
.q
