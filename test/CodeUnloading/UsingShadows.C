//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S -Xclang -verify 2>&1 | FileCheck %s

extern "C" int printf(const char*, ...);

namespace C { struct foo { foo() { printf("foo\n"); } }; }
// FIXME .compareState X fails when a runtime function is called
// after .storeState X has been run
C::foo();
// CHECK: foo
.storeState "C"

namespace D {}
.storeState "CD"

namespace D { using C::foo; }
namespace D { using C::foo; }
.storeState "CDD"

namespace D { using C::foo; }
namespace D { using C::foo; }
.undo
.undo

.compareState "CDD"
//CHECK-NOT: Differences


namespace D { using C::foo; }
namespace D { using C::foo; }
namespace D { using C::foo; }
namespace D { using C::foo; }
.undo
.undo
.undo
.undo
.compareState "CDD"
//CHECK-NOT: Differences

D::foo();
// CHECK-NEXT: foo

.undo
.undo
.undo
.compareState "CD"
//CHECK-NOT: Differences

D::foo(); //expected-error {{no member named 'foo' in namespace 'D'}}

.undo
.compareState "C"
//CHECK-NOT: Differences

namespace A { void foo(); }
.storeState "TESTU1"
namespace B { using A::foo; }
.undo
.compareState "TESTU1"
// CHECK-NOT: Differences

.storeState "TESTU2"
namespace B { using A::foo; }
.undo
.compareState "TESTU2"
// CHECK-NOT: Differences

.storeState "TESTU3"
namespace B { using A::foo; }
.undo

.compareState "TESTU3"
// CHECK-NOT: Differences

typedef long long_t;

.storeState "TEST4"

#include "UsingShadows.h"
#include "UsingShadows.h"
#include "UsingShadows.h"
.undo
.undo
.undo

//Make sure long_t is still valid
.compareState "TEST4"
// CHECK-NOT: Differences

long_t val = 9;
val
// CHECK-NEXT: 9

// Unloading <string> used to fail as well (which as annoying).
#include <string>
.undo

.q
