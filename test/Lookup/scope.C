//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -fno-rtti 2>&1 | FileCheck %s
// Test findScope, which is essentially is a DeclContext.
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"

#include <cstdio>

using namespace std;
using namespace llvm;
using namespace cling;

.rawInput 1
class A {};
.rawInput 0

const LookupHelper& lookup = gCling->getLookupHelper();
LookupHelper::DiagSetting diags = LookupHelper::WithDiagnostics;

const clang::Decl* cl_A = lookup.findScope("A", diags);
printf("cl_A: 0x%lx\n", (unsigned long) cl_A);
//CHECK: cl_A: 0x{{[1-9a-f][0-9a-f]*$}}
cast<clang::NamedDecl>(cl_A)->getQualifiedNameAsString().c_str()
//CHECK-NEXT: ({{[^)]+}}) "A"

.rawInput 1
namespace N {
class A {};
}
.rawInput 0


const clang::Decl* cl_A_in_N = lookup.findScope("N::A", diags);
printf("cl_A_in_N: 0x%lx\n", (unsigned long) cl_A_in_N);
//CHECK: cl_A_in_N: 0x{{[1-9a-f][0-9a-f]*$}}
cast<clang::NamedDecl>(cl_A_in_N)->getQualifiedNameAsString().c_str()
//CHECK-NEXT: ({{[^)]+}}) "N::A"


.rawInput 1
namespace N {
namespace M {
namespace P {
class A {};
}
}
}
.rawInput 0

const clang::Decl* cl_A_in_NMP = lookup.findScope("N::M::P::A", diags);
cl_A_in_NMP
//CHECK: (const clang::Decl *) 0x{{[1-9a-f][0-9a-f]*$}}
cast<clang::NamedDecl>(cl_A_in_NMP)->getQualifiedNameAsString().c_str()
//CHECK-NEXT: ({{[^)]+}}) "N::M::P::A"


.rawInput 1
template <class T> class B { T b; };
.rawInput 0

const clang::Decl* cl_B_int = lookup.findScope("B<int>", diags);
printf("cl_B_int: 0x%lx\n", (unsigned long) cl_B_int);
//CHECK-NEXT: cl_B_int: 0x{{[1-9a-f][0-9a-f]*$}}



//
//  Test optional returned type is as spelled by the user.
//

typedef int Int_t;

.rawInput 1
template <typename T> struct W { T member; };
.rawInput 0

const clang::Type* resType = 0;
lookup.findScope("W<Int_t>", diags, &resType);
//resType->dump();
clang::QualType(resType,0).getAsString().c_str()
//CHECK-NEXT: ({{[^)]+}}) "W<Int_t>"

.rawInput 1
namespace Functions {
void Next();
}

using namespace Functions;

namespace Next {}
.rawInput 0

const clang::Decl* cl_Next = lookup.findScope("Next", diags);
printf("cl_Next: 0x%lx\n", (unsigned long) cl_Next);
//CHECK-NEXT: cl_Next: 0x{{[1-9a-f][0-9a-f]*$}}
