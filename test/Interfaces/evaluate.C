// RUN: cat %s | %cling -Xclang -verify
// RUN: cat %s | %cling | FileCheck %s

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/StoredValueRef.h"

cling::StoredValueRef V;
V // CHECK: (cling::StoredValueRef) <<<invalid>>> @0x{{.*}}

gCling->evaluate("1;", V);
V // CHECK: (cling::StoredValueRef) boxes [(int) 1]

long LongV = 17;
gCling->evaluate("LongV;", V);
V // CHECK: (cling::StoredValueRef) boxes [(long) 17]

int* IntP = (int*)0x12;
gCling->evaluate("IntP;", V);
V // CHECK: (cling::StoredValueRef) boxes [(int *) 0x12]

cling::StoredValueRef Result;
gCling->evaluate("V", Result);
Result // CHECK: (cling::StoredValueRef) boxes [(cling::StoredValueRef)]
V // CHECK: (cling::StoredValueRef) boxes [(int *) 0x12]

// Savannah #96277
gCling->evaluate("double sin(double); double one = sin(3.141/2);", V);
V // CHECK: (cling::StoredValueRef) boxes [(double) 1.000000e+00]

gCling->process("double sin(double); double one = sin(3.141/2);", &V);
V // CHECK: (cling::StoredValueRef) boxes [(double) 1.000000e+00]
one // CHECK: (double) 1.000
int one; // expected-error {{redefinition of 'one' with a different type: 'int' vs 'double'}} expected-note {{previous definition is here}}

// Make sure that PR#98434 doesn't get reintroduced.
void f(int);
gCling->evaluate("f // expected-error {{cannot initialize return object of type 'void (int)' with an lvalue of type 'void (int)'}}", V);
// end PR#98434
