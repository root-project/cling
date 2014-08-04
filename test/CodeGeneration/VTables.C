//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify | FileCheck %s
// XFAIL:*

// Test whether the interpreter is able to generate properly the symbols
// and the vtables of classes.

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Utils/AST.h"
#include "clang/AST/Decl.h"
#include "llvm/Support/Casting.h"
.rawInput

template <typename T> struct C {
   virtual void f() {}
   static int S;
};
template<typename T> int C<T>::S = 12;
template class C<int>;

// Nested in function class
void f() {
  class NestedC {
  public:
    virtual void g() {}
  };
}

.rawInput

const cling::LookupHelper& lh = gCling->getLookupHelper();
clang::Sema& S = gCling->getSema();
const clang::NamedDecl* D = 0;

clang::DeclContext* DC = 0;
DC = llvm::dyn_cast_or_null<clang::DeclContext>(const_cast<clang::Decl*>(lh.findScope("C<int>")));
D = cling::utils::Lookup::Named(&S, "S", DC);
gCling->getAddressOfGlobal(D)
//CHECK-NOT: (void *) 0x0

D = cling::utils::Lookup::Named(&S, "f", DC);
gCling->getAddressOfGlobal(D)
//TODO-CHECK-NOT: (void *) 0x0

.rawInput
// Nested classes
class N1 {
public:
  class N2 {
  public:
    class N3 {
    public:
      virtual void fN3() {}
    };
    virtual void fN2() = 0;
  };
};
.rawInput

DC = llvm::dyn_cast_or_null<clang::DeclContext>(const_cast<clang::Decl*>(lh.findScope("N1::N2")));
D = cling::utils::Lookup::Named(&S, "fN2", DC);
gCling->getAddressOfGlobal(D)
//CHECK-NOT: (void *) 0x0

.rawInput

// vbases
class V { public: virtual void fV() {} };
class B1 : virtual public V { /* ... */ };
class B2 : virtual public V { /* ... */ };
class B3 : public V { /* ... */ };
class X : public B1, public B2, public B3 {
public:
  virtual void fV(){}
  struct S {
    virtual void fS() {}
  };
};

.rawInput

DC = llvm::dyn_cast_or_null<clang::DeclContext>(const_cast<clang::Decl*>(lh.findScope("X")));
D = cling::utils::Lookup::Named(&S, "fV", DC);
gCling->getAddressOfGlobal(D)
//CHECK-NOT: (void *) 0x0

DC = llvm::dyn_cast_or_null<clang::DeclContext>(const_cast<clang::Decl*>(lh.findScope("X::S")));
D = cling::utils::Lookup::Named(&S, "fS", DC);
gCling->getAddressOfGlobal(D)
//CHECK-NOT: (void *) 0x0
