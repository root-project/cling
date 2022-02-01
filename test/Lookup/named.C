//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -fno-rtti 2>&1 | FileCheck %s
// Test Lookup::Named and Namespace, used in quick simple lookups.

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Utils/AST.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"

#include <cstdio>

using namespace cling;
using namespace llvm;

.rawInput 1

namespace Functions {
void Next();
}

using namespace Functions;

namespace Next {
class Inside_Next {};
}

namespace AnotherNext {
class Inside_AnotherNext {};
}

.rawInput 0

// ROOT-6095: names introduced in a scopeless enum should be available in the
// parent context.
typedef enum { k0, k1 } E;
E foo = k1
//CHECK: (E) (k1) : (unsigned int) 1
struct X { enum { k0, k1 = 2 }; } bar
X::k1
//CHECK: (X::(anonymous)) (X::k1) : (unsigned int) 2

clang::Sema& S = gCling->getSema();
const LookupHelper& lookup = gCling->getLookupHelper();
LookupHelper::DiagSetting diags = LookupHelper::WithDiagnostics;

const clang::NamedDecl *decl{nullptr};

decl = utils::Lookup::Named(&S, "AnotherNext", nullptr);
decl
//CHECK: (const clang::NamedDecl *) 0x{{[1-9a-f][0-9a-f]*$}}
decl->getQualifiedNameAsString().c_str()
//CHECK-NEXT: ({{[^)]+}}) "AnotherNext"

const clang::DeclContext *context = dyn_cast<clang::DeclContext>(decl);
context
//CHECK: (const clang::DeclContext *) 0x{{[1-9a-f][0-9a-f]*$}}

decl = utils::Lookup::Named(&S, "Inside_AnotherNext", context);
decl
//CHECK: (const clang::NamedDecl *) 0x{{[1-9a-f][0-9a-f]*$}}
decl->getQualifiedNameAsString().c_str()
//CHECK-NEXT: ({{[^)]+}}) "AnotherNext::Inside_AnotherNext"

decl = utils::Lookup::Named(&S, "k1", nullptr);
decl
//CHECK: (const clang::NamedDecl *) 0x{{[1-9a-f][0-9a-f]*$}}

// Now test the ambiguities.

decl = utils::Lookup::Named(&S, "Next", nullptr);
decl
//CHECK: (const clang::NamedDecl *) 0x{{f+[ $]}}


const clang::Decl* nextDecl = lookup.findScope("Next", diags);
nextDecl
//CHECK: (const clang::Decl *) 0x{{[1-9a-f][0-9a-f]*$}}
cast<clang::NamedDecl>(nextDecl)->getQualifiedNameAsString().c_str()
//CHECK-NEXT: ({{[^)]+}}) "Next"

context = llvm::dyn_cast<clang::DeclContext>(nextDecl);
context
//CHECK: (const clang::DeclContext *) 0x{{[1-9a-f][0-9a-f]*$}}

decl = utils::Lookup::Named(&S, "Inside_Next", context);
decl
//CHECK: (const clang::NamedDecl *) 0x{{[1-9a-f][0-9a-f]*$}}

// Now test looking up non-existing things.

// In global scope
decl = utils::Lookup::Named(&S, "DoesNotExist", nullptr);
decl
//CHECK: (const clang::NamedDecl *) {{0x0*$|nullptr}}

// In a namespace
decl = utils::Lookup::Named(&S, "EvenLess", context);
decl
//CHECK: (const clang::NamedDecl *) {{0x0*$|nullptr}}
