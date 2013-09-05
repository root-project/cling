// RUN: cat %s | %cling -Xclang -verify
// RUN: cat %s | %cling | FileCheck %s

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/StoredValueRef.h"

cling::StoredValueRef V;
V // CHECK: (cling::StoredValueRef) <<<invalid>>> @0x{{.*}}

gCling->evaluate("return 1;", V);
V // CHECK: (cling::StoredValueRef) boxes [(int) 1]

long LongV = 17;
gCling->evaluate("LongV;", V);
V // CHECK: (cling::StoredValueRef) boxes [(long) 17]

int* IntP = (int*)0x12;
gCling->evaluate("IntP;", V);
V // CHECK: (cling::StoredValueRef) boxes [(int *) 0x12]

cling::StoredValueRef Result;
gCling->evaluate("V", Result);
// Here we check what happens for record type like cling::StoredValueRef; they are returned by reference.
Result // CHECK: (cling::StoredValueRef) boxes [(cling::StoredValueRef &) &0x{{.*}}]
V // CHECK: (cling::StoredValueRef) boxes [(int *) 0x12]

// Savannah #96277
gCling->evaluate("double sin(double); double one = sin(3.141/2);", V);
V // CHECK: (cling::StoredValueRef) boxes [(double) 1.000000e+00]

gCling->process("double sin(double); double one = sin(3.141/2);", &V);
V // CHECK: (cling::StoredValueRef) boxes [(double) 1.000000e+00]
one // CHECK: (double) 1.000
int one; // expected-error {{redefinition of 'one' with a different type: 'int' vs 'double'}} expected-note {{previous definition is here}}

// Make sure that PR#98434 doesn't get reintroduced.
.rawInput
void f(int) { return; }
.rawInput

gCling->evaluate("f", V);
V.isValid() //CHECK: {{\([_]B|b}}ool) true
// end PR#98434
