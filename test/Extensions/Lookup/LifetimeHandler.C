// RUN: cat %s | %cling -I%p | FileCheck %s
// We should revise the destruction of the LifetimeHandlers, because
// its destructor uses gCling and the CompilerInstance, which are 
// already gone

#include "cling/Interpreter/InterpreterCallbacks.h"

.dynamicExtensions 1

.x LifetimeHandler.h
// CHECK: Alpha's single arg ctor called {{.*Interpreter.*}}
// CHECK: After Alpha is Beta {{.*Interpreter.*}}
// CHECK: Alpha dtor called {{.*Interpreter.*}}

Alpha a(sadasds->getVersion()); 
printf("%s\n", a.getVar()); // CHECK: {{.*Interpreter.*}}

int res = h->Add10(h->Add10(h->Add10(0))) // CHECK: (int) 30

.q
