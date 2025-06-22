//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p -Xclang -verify 2>&1

// Test failures of dynamic lookups.

// This test tests the hook that cling expects in clang and enables it
// at compile time. However that is not actual dynamic lookup because
// the compiler uses the hook at compile time, since we enable it on
// creation. When a symbol lookup fails in compile time, clang asks its
// externally attached sources whether they could provide the declaration
// that is being lookedup. In the current test our external source is enabled
// so it provides it on the fly, and no transformations to delay the lookup
// are performed.

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"

.dynamicExtensions 1

std::unique_ptr<cling::test::SymbolResolverCallback> SRC;
SRC.reset(new cling::test::SymbolResolverCallback(gCling, false))
gCling->setCallbacks(std::move(SRC));
p.q
// expected-error@input_line_28:2 {{use of undeclared identifier 'p'}}

// `auto` is not supported
auto x = unknownexpression()
// expected-error@input_line_31:2 {{cannot deduce 'auto' from unknown expression}}
.q
