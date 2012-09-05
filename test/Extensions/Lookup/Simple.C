// RUN: cat %s | %cling -I%p | FileCheck %s

// This test tests the hook that cling expects in clang and enables it
// at compile time. However that is not actual dynamic lookup because
// the compiler uses the hook at compile time, since we enable it on
// creation. When a symbol lookup fails in compile time, clang asks its
// externally attached sources whether they could provide the declaration
// that is being lookedup. In the current test our external source is enabled
// so it provides it on the fly, and no transformations to delay the lookup
// are performed.

#include "SymbolResolverCallback.h"

.dynamicExtensions

gCling->setCallbacks(new cling::test::SymbolResolverCallback(gCling, /*Enabled=*/true));
jksghdgsjdf->getVersion() // CHECK: {{.*Interpreter.*}}
hsdghfjagsp->Draw() // CHECK: (int const) 12

h->Add10(h->Add10(h->Add10(0))) // CHECK: (int const) 30
h->PrintString(std::string("test")); // CHECK: test
int a[5] = {1,2,3,4,5};
h->PrintArray(a, 5); // CHECK: 12345
.q
