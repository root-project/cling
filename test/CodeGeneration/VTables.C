// RUN: cat %s | %cling | FileCheck %s
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

.rawInput

const cling::LookupHelper& lh = gCling->getLookupHelper();
const clang::NamedDecl* D = 0;

clang::DeclContext* DC = 0;
DC = llvm::dyn_cast_or_null<clang::DeclContext>(const_cast<clang::Decl*>(lh.findScope("C<int>")));
D = cling::utils::Lookup::Named(&gCling->getSema(), "S", DC);
gCling->getAddressOfGlobal(D)
//CHECK-NOT: (void *) 0x0

D = cling::utils::Lookup::Named(&S, "f", DC);
gCling->getAddressOfGlobal(D)
//TODO-CHECK-NOT: (void *) 0x0
