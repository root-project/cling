//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

// Actually test clang::DeclContext::removeDecl(). This function in clang is
// the main method that is used for the error recovery. This means when there
// is an error in cling's input we need to revert all the declarations that came
// in from the same transaction. Even when we have anonymous declarations we
// need to be able to remove them from the declaration context. In a compiler's
// point of view there is no way that one can call removeDecl() and pass in anon
// decl, because the method is used when shadowing decls, which must have names.
// The issue is (and we patched it) is that removeDecl is trying to remove the
// anon decl (which doesn't have name) from the symbol (lookup) tables, which
// doesn't make sense.
// The current test checks if that codepath in removeDecl still exists because
// it is important for the stable error recovery in cling

.storeState "testMyClass"
class MyClass {
  struct {
    int a;
    error_here; // expected-error {{C++ requires a type specifier for all declarations}}
  };
};
.compareState "testMyClass"
 // CHECK-NOT: File with AST differencies stored in: testMyClassAST.diff

.storeState "testStructX"
struct X {
  union {
    float f3;
    double d2;
  } named;

  union {
    int i;
    float f;

    union {
      float f2;
      mutable double d;
    };
  };

  void test_unqual_references();

  struct {
    int a;
    float b;
  };

  void test_unqual_references_const() const;

  mutable union { // expected-error{{anonymous union at class scope must not have a storage specifier}}
    float c1;
    double c2;
  };
};
.compareState "testStructX"
// CHECK-NOT: File with AST differencies stored in: testStructXAST.diff
// Make FileCheck happy with having at least one positive rule:
int a = 5
// CHECK: (int) 5
.q

