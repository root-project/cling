//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

// This file should be used as regression test for the value printing subsystem
// Reproducers of fixed bugs should be put here

int Var = 43;

auto LD = []  { return Var; };
auto LR = [&] { return Var; };

auto LC = [=] { return Var; } // expected-warning {{captures will be by reference, not copy}}
// CHECK: ((lambda) &) @0x{{.*}}

++Var;
LC()
// CHECK-NEXT: (int) 44

LD() == LR() && LD() == LC()
// CHECK-NEXT: (bool) true

auto LL = [=,&Var] { return Var; };
// expected-error@input_line_24:2 {{'Var' cannot be captured because it does not have automatic storage duration}}
// expected-note@input_line_13:2 {{'Var' declared here}}
// expected-warning@input_line_24:2 {{captures will be by reference, not copy}}

.q
