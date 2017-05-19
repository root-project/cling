//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S -Xclang -verify 2>&1 | FileCheck %s
// Test inlineNamespaces

#define TEST_NAMESPACE test
#include "NameSpaces.h"
.undo
#include "NameSpaces.h"
.undo
#undef TEST_NAMESPACE

#define TEST_NAMESPACE std
#include "NameSpaces.h"
.undo
#include "NameSpaces.h"
.undo
#undef TEST_NAMESPACE


namespace A{}
namespace A { inline namespace __BBB { int f; } }
namespace A { inline namespace __BBB { int f1; } }
namespace A { inline namespace __BBB { int f2; } }
.undo
A::f2 // expected-error {{no member named 'f2' in namespace 'A'}}
.undo
A::f1 // expected-error {{no member named 'f1' in namespace 'A'}}
.undo
A::f // expected-error {{no member named 'f' in namespace 'A'}}


#include <stdexcept>
.undo
#include <stdexcept>
.undo

101
// CHECK: (int) 101

.q
