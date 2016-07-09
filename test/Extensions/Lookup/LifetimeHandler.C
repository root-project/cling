//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %built_cling -I%p | FileCheck %s
// We should revise the destruction of the LifetimeHandlers, because
// its destructor uses gCling and the CompilerInstance, which are
// already gone

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"

.dynamicExtensions 1

std::unique_ptr<cling::test::SymbolResolverCallback> SRC;
SRC.reset(new cling::test::SymbolResolverCallback(gCling))
gCling->setCallbacks(std::move(SRC));

.x LifetimeHandler.h
// CHECK: Alpha's single arg ctor called {{.*Interpreter.*}}
// CHECK: After Alpha is Beta {{.*Interpreter.*}}
// CHECK: Alpha dtor called {{.*Interpreter.*}}

Alpha a(sadasds->getVersion());
printf("%s\n", a.getVar()); // CHECK: {{.*Interpreter.*}}

int res = h->Add10(h->Add10(h->Add10(0))) // CHECK: (int) 30

.q
