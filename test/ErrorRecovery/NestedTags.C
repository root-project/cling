//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: true

//

FIX RUN LINE BELOW TO TURN THE TEST BACK ON!!!
Currently, builtins cannot reliably ignored in the comparison of before and
  after, causing this test to sometimes fail.

// : cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

// Tests the removal of nested decls
.storeState "testNestedDecls1"
struct Outer { struct Inner { enum E{i = 1}; }; };error_here; // expected-error {{use of undeclared identifier 'error_here'}}
.compareState "testNestedDecls1"
// CHECK-NOT: Differences

.rawInput 1
.storeState "parentLookupTables"
enum AnEnum {
   aa = 3,
   bb = 4,
 }; error_here; // expected-error {{C++ requires a type specifier for all declarations}}
.compareState "parentLookupTables"
.rawInput 0
// CHECK-NOT: Differences

.rawInput
namespace Outer { struct Inner { enum E{i = 2}; }; };
.rawInput

enum A{a};

.storeState "testNestedDecls2"
enum A{a}; // expected-error {{redefinition of 'A'}} expected-note {{previous definition is here}}
.compareState "testNestedDecls2"
// CHECK-NOT: Differences

Outer::Inner::i
// CHECK: (Outer::Inner::E::i) : (int) 2

a // CHECK: (enum A) (A::a) : (int) 0
.q
