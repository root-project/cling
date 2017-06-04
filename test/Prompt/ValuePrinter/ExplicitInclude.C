//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.

//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

#include "cling/Interpreter/RuntimePrintValue.h"

struct Trigger { } trgr
// CHECK: (struct Trigger &) @0x{{.*}}
.undo

struct Trigger2 { } trgr
// CHECK-NEXT: (struct Trigger2 &) @0x{{.*}}
.undo

.undo // #include "cling/Interpreter/RuntimePrintValue.h"

struct Trigger3 { } trgr
// CHECK-NEXT: (struct Trigger3 &) @0x{{.*}}

// expected-no-diagnostics
.q
