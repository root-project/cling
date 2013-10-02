// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

// Tests the removal of nested decls 

.storeState "testNestedDecls1"
struct Outer { struct Inner { enum E{i = 1}; }; };error_here; // expected-error {{use of undeclared identifier 'error_here'}}
.compareState "testNestedDecls1"

.rawInput
namespace Outer { struct Inner { enum E{i = 2}; }; }; 
.rawInput

Outer::Inner::i
// CHECK: (Outer::Inner::E::i) : (int) 2

enum A{a};

.storeState "testNestedDecls2"
enum A{a}; // expected-error {{redefinition of 'A'}} expected-note {{previous definition is here}}
.compareState "testNestedDecls2"
// CHECK-NOT: Differences
a // CHECK: (enum A) (A::a) : (int) 0
.q
