//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling --noruntime -I%S -Xclang -verify 2>&1 | FileCheck --allow-empty %s
// Test unloadMath

// grab all the builtins we can
#include <math.h>
.undo
// used to be a problem unloading here

.storeState "A"
#include <math.h>
#include <stddef.h>
.undo
.undo
.compareState "A"
// nullptr_t using declaration used to hang around
// CHECK-NOT: Differences

// expected-no-diagnostics
.q
