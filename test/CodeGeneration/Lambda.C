//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s

// Check that lambda mangling is stable across transactions (ROOT-10689)

.rawInput 1
extern "C" int printf(const char*,...);
template <class F> void call(F &f) { f(); }
auto l1 = []() -> int { printf("ONE\n"); return 42; }; auto l2 = []() -> long { printf("TWO\n"); return 17; };
.rawInput 0
call(l1); call(l2); // CHECK: ONE
// CHECK-NEXT: TWO
call(l2);
// CHECK-NEXT: TWO
