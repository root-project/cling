//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: clang -shared %S/address_lib.c -olibaddress_lib%shlibext
// RUN: cat %s | %cling -L. -fno-rtti | FileCheck %s
extern "C" int printf(const char*,...);

#include "cling/Interpreter/Interpreter.h"
#include "cling/Utils/AST.h"
#include "clang/AST/Decl.h"
#include "clang/AST/GlobalDecl.h"
.rawInput
const char* comp(void* A, void* B) {
  if (A == B) { return "equal"; }
  else { printf("[%p %p] ", A, B); return "DIFFER!"; }
}
.rawInput

bool fromJIT = false;
clang::Sema& sema = gCling->getSema();
int gMyGlobal = 12;
void* addr1 = &gMyGlobal;
clang::NamedDecl* D = cling::utils::Lookup::Named(&sema, "gMyGlobal");
if (!D) printf("gMyGlobal decl not found!\n");
clang::VarDecl* VD = llvm::cast<clang::VarDecl>(D);
void* addr2 = gCling->getAddressOfGlobal(clang::GlobalDecl(VD), &fromJIT);
if (!fromJIT) printf("gMyGlobal should come from JIT!\n");
printf("gMyGlobal: %s\n", comp(addr1, addr2)); // CHECK: gMyGlobal: equal

.rawInput
namespace N {
   int gMyGlobal = 13;
}
.rawInput
void* addrN1 = &N::gMyGlobal;
clang::NamespaceDecl* ND = cling::utils::Lookup::Namespace(&sema, "N");
if (!ND) printf("namespace N decl not found!\n");
fromJIT = false;
VD = llvm::cast<clang::VarDecl>(cling::utils::Lookup::Named(&sema, "gMyGlobal", ND));
if (!VD) printf("N::gMyGlobal decl not found!\n");
void* addrN2 = gCling->getAddressOfGlobal(clang::GlobalDecl(VD), &fromJIT);
if (!fromJIT) printf("N::gMyGlobal should come from JIT!\n");
printf("N::gMyGlobal: %s\n", comp(addrN1, addrN2)); //CHECK: N::gMyGlobal: equal

.L address_lib
extern "C" int gLibGlobal;
void* addrL1 = &gLibGlobal;
fromJIT = true;
VD = llvm::cast<clang::VarDecl>(cling::utils::Lookup::Named(&sema, "gLibGlobal"));
if (!VD) printf("gLibGlobal decl not found!\n");
void* addrL2 = gCling->getAddressOfGlobal(clang::GlobalDecl(VD), &fromJIT);
if (fromJIT) printf("gLibGlobal should NOT come from JIT!\n");
printf("gLibGlobal: %s\n", comp(addrL1, addrL2)); //CHECK: gLibGlobal: equal
