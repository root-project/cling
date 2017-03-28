//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling --noruntime -I%S -Xclang -verify 2>&1
// RUN: cat %s | %cling -I%S -Xclang -verify  2>&1 | FileCheck %s -allow-empty
// Test nested-namespace


namespace A { }
.storeState "A"
namespace A { inline namespace __1 {} }
.storeState "__1"
namespace A { inline namespace __1 { inline namespace __2 {} } }
namespace A { inline namespace __1 { inline namespace __2 { struct T {}; }} }

.undo
.undo
.compareState "__1"
// CHECK-NOT: Differences
        
namespace A { inline namespace __1 { namespace __2 {} } }
namespace A { inline namespace __1 { namespace __2 { struct T {}; }} }
.undo
.undo
.compareState "__1"

.undo
.compareState "A"
// CHECK-NOT: Differences

namespace A { namespace __1 {} }
.storeState "__1"
namespace A { namespace __1 { inline namespace __2 {} } }
namespace A { namespace __1 { inline namespace __2 { struct T {}; } } }

.undo
.undo
.compareState "__1"
// CHECK-NOT: Differences

.undo
.compareState "A"
// CHECK-NOT: Differences

// expected-no-diagnostics
.q
