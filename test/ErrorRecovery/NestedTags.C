// RUN: cat %s | %cling -Xclang -verify -I%p | FileCheck %s

// Tests the removal of nested decls 

struct Outer { struct Inner { enum E{i = 1}; }; };error_here; // expected-error {{error: use of undeclared identifier 'error_here'}}

.rawInput
namespace Outer { struct Inner { enum E{i = 2}; }; }; 
.rawInput

Outer::Inner::i
// CHECK: (enum Outer::Inner::E const) @0x{{[0-9A-Fa-f]{7,12}.}} 
// CHECK: (Outer::Inner::E::i) : (int) 2

enum A{a}; // 
enum A{a}; // expected-error {{redefinition of 'A'}} expected-note {{previous definition is here}}
a // expected-error {{use of undeclared identifier 'a'}}

.q
