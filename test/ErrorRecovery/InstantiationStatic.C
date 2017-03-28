//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// This currently fails becuase TestStatic<> is fully instantiated due to
// the static variable initialization.
// TestStatic<> now references TestIter<int> (A).
// After TestIterInst<> failure, TestIter<> (A) is unloaded, then later
// re-instantiated as TestIter<> (B).
// When 'for (auto val : ts0 )' is compiled it is expecting a TestIter<> (B)
// but 'TestStatic<>::begin/end' are still returning a TestIter<> (A)
//
// Later uses will work; however, as the bad failure causes begin and end on
// TestStatic<> to be reinstantiated with TestIter<> (B).

// Test instantiationRecover
// RUN: cat %s | %cling -I %S -Xclang -verify 2>&1 | FileCheck %s

extern "C" int printf(const char*,...);
#include "Instantiation.h"

TestIterInst<int> ti;
for (auto val : ti.test1(12) ) { val(); }
// expected-error@2 {{called object type 'int' is not a function or function pointer}}

TestStatic<int> ts0;
for (auto val : ts0 ) { printf("{ %d }\n", val); }
// CHECK: { 0 }

.q
