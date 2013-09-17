// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

// Tests the removal of nested decls 

.storeState "testNestedDecls1"
struct Outer { struct Inner { enum E{i = 1}; }; };error_here; // expected-error {{error: use of undeclared identifier 'error_here'}}
.compareState "testNestedDecls1"
// CHECK-NOT: File with AST differencies stored in: testNestedDeclsAST1.diff

.rawInput
namespace Outer { struct Inner { enum E{i = 2}; }; }; 
.rawInput

Outer::Inner::i
// CHECK: (Outer::Inner::E::i) : (int) 2

enum A{a}; // 

.storeState "testNestedDecls2"
enum A{a}; // expected-error {{redefinition of 'A'}} expected-note {{previous definition is here}}
a // expected-error {{use of undeclared identifier 'a'}}
.compareState "testNestedDecls2"
// CHECK-NOT: File with AST differencies stored in: testNestedDeclsAST2.diff
.q
 
