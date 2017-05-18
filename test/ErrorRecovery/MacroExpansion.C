//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

#define BEGIN_NAMESPACE namespace test_namespace {
#define END_NAMESPACE }

.rawInput 1

BEGIN_NAMESPACE int j; END_NAMESPACE // expected-note {{previous definition is here}}

.storeState "testMacroExpansion"
BEGIN_NAMESPACE int j; END_NAMESPACE // expected-error {{redefinition of 'j'}}
.compareState "testMacroExpansion"
.rawInput 0
// CHECK-NOT: Differences
// Make FileCheck happy with having at least one positive rule:
int a = 5
// CHECK: (int) 5
.q
