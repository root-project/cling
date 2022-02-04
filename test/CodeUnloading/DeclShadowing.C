//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s
#include "cling/Interpreter/Interpreter.h"
#include "cling/Utils/AST.h"
#include "clang/AST/Decl.h"

#include <type_traits>
#include <cstdlib>
#include <string>

unsigned _i;
struct _X {};
int _g() { return 1; }

cling::runtime::gClingOpts->AllowRedefinition = 1;

void g() {}

// ==== Test UsingDirectiveDecl/UsingDecl
// These should not be nested into a `__cling_N5xxx' namespace (but placed at
// the TU scope) so that the declaration they name is globally available.
using namespace std;
namespace NS { string baz("Cling"); }
using NS::baz;
baz
//CHECK: (std::string &) "Cling"

// ==== Redeclare `i' with a different type
int i = 1
//CHECK-NEXT: (int) 1
double i = 3.141592
//CHECK-NEXT: (double) 3.1415920

// ==== Provide different definition for type
struct MyStruct;
struct MyStruct { unsigned u; };
MyStruct{1111U}.u
//CHECK-NEXT: (unsigned int) 1111
struct MyStruct { int i, j; } foo{33, 0}
//CHECK-NEXT: (struct MyStruct &)
foo.i
//CHECK-NEXT: (int) 33
struct MyStruct;  // nop
MyStruct{0, 99}.j
//CHECK-NEXT: (int) 99

// ==== Make `foo' a typedef type
typedef int foo;
foo bar = 1
//CHECK-NEXT: (int) 1
typedef MyStruct foo;
foo{11, 99}.i
//CHECK-NEXT: (int) 11
bar
// CHECK-NEXT: (int) 1

// ==== Give a new defintion for `foo'; test function overload
char foo(int x) { return 'X'; }
int foo() { return 0; }
foo()
//CHECK-NEXT: (int) 0
foo(0)
//CHECK-NEXT: (char) 'X'
double foo() { return 1; }
foo()
//CHECK-NEXT: (double) 1.0000000

// ==== (Re)define a class template
template <typename T> struct S { T i; };
S<int> foo{99};
foo.i
//CHECK-NEXT: (int) 99
template <typename T> struct S { T i, j; };
S<double>{0, 33.0}.j
//CHECK-NEXT: (double) 33.000000

// ==== Test function templates
template <typename T>
typename std::enable_if<std::is_same<T, int>::value, int>::type f(T x) { return x; }
template <typename T>
typename std::enable_if<!std::is_same<T, int>::value, int>::type f(T x) { return 0x55aa; }
f(33)
//CHECK-NEXT: (int) 33
f(3.3f)
//CHECK-NEXT: (int) 21930

template <typename T>
typename std::enable_if<std::is_same<T, int>::value, int>::type f(T x) { return 0xaa55; }
f(33)
//CHECK-NEXT: (int) 43605
f(3.3f)
//CHECK-NEXT: (int) 21930

cling::runtime::gClingOpts->AllowRedefinition = 0;

// ==== Check DeclContext
// If DefinitionShadower is enabled, declaration context is an inline namespace;
// otherwise it should be the TU.
clang::Sema& _S = gCling->getSema();
const clang::NamedDecl *_decl{nullptr};

_decl = cling::utils::Lookup::Named(&_S, "_i", nullptr)
//CHECK-NEXT: (const clang::NamedDecl *) 0x{{[1-9a-f][0-9a-f]*$}}
_decl->getDeclContext()->isTranslationUnit()
//CHECK-NEXT: (bool) true

_decl = cling::utils::Lookup::Named(&_S, "_X", nullptr)
//CHECK-NEXT: (const clang::NamedDecl *) 0x{{[1-9a-f][0-9a-f]*$}}
_decl->getDeclContext()->isTranslationUnit()
//CHECK-NEXT: (bool) true

_decl = cling::utils::Lookup::Named(&_S, "_g", nullptr)
//CHECK-NEXT: (const clang::NamedDecl *) 0x{{[1-9a-f][0-9a-f]*$}}
_decl->getDeclContext()->isTranslationUnit()
//CHECK-NEXT: (bool) true

_decl = cling::utils::Lookup::Named(&_S, "bar", nullptr)
//CHECK-NEXT: (const clang::NamedDecl *) 0x{{[1-9a-f][0-9a-f]*$}}
_decl->getDeclContext()->isInlineNamespace()
//CHECK-NEXT: (bool) true

_decl = cling::utils::Lookup::Named(&_S, "MyStruct", nullptr)
//CHECK-NEXT: (const clang::NamedDecl *) 0x{{[1-9a-f][0-9a-f]*$}}
_decl->getDeclContext()->isInlineNamespace()
//CHECK-NEXT: (bool) true

_decl = cling::utils::Lookup::Named(&_S, "g", nullptr)
//CHECK-NEXT: (const clang::NamedDecl *) 0x{{[1-9a-f][0-9a-f]*$}}
_decl->getDeclContext()->isInlineNamespace()
//CHECK-NEXT: (bool) true

//expected-no-diagnostics
.q
