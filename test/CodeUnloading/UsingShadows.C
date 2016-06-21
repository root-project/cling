//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S 2>&1 | FileCheck %s

namespace A { void foo(); }
.storeState "TESTU1"
namespace B { using A::foo; }
.undo
.compareState "TESTU1"
// CHECK-NOT: Differences

.storeState "TESTU2"
namespace B { using A::foo; }
.undo
.compareState "TESTU2"
// CHECK-NOT: Differences

.storeState "TESTU3"
namespace B { using A::foo; }
.undo

.compareState "TESTU3"
// CHECK-NOT: Differences

typedef long long_t;

.storeState "TEST4"

#include "UsingShadows.h"
#include "UsingShadows.h"
#include "UsingShadows.h"
.undo
.undo
.undo

//Make sure long_t is still valid
.compareState "TEST4"
// CHECK-NOT: Differences

long_t val = 9;
val
// CHECK: 9

// Unloading <string> used to fail as well (which as annoying).
#include <string>
.undo

// expected-no-diagnostics
.q
