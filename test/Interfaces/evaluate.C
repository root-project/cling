//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: test "x`uname -m|sed 's,i.86,i386,'`" = "xi386" || cat %s | %cling -Xclang -verify | FileCheck %s
// Not running on 32 bit due to aggregate return in getWithDtor(); see ROOT-5860

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/StoredValueRef.h"

cling::StoredValueRef V;
V // CHECK: (cling::StoredValueRef) <<<invalid>>> @0x{{.*}}

gCling->evaluate("return 1;", V);
V // CHECK: (cling::StoredValueRef) boxes [(int) 1]

// Returns must put the result in the StoredValueRef.
bool cond = true;
gCling->evaluate("if (cond) return \"true\"; else return 0;", V);
V // CHECK: (cling::StoredValueRef) boxes [(const char [5]) "true"]
gCling->evaluate("cond = false; if (cond) return \"true\"; else return 0;", V);
V // CHECK: (cling::StoredValueRef) boxes [(int) 0]

long LongV = 17;
gCling->evaluate("LongV;", V);
V // CHECK: (cling::StoredValueRef) boxes [(long) 17]

int* IntP = (int*)0x12;
gCling->evaluate("IntP;", V);
V // CHECK: (cling::StoredValueRef) boxes [(int *) 0x12]

cling::StoredValueRef Result;
gCling->evaluate("V", Result);
// Here we check what happens for record type like cling::StoredValueRef; they are returned by reference.
Result // CHECK: (cling::StoredValueRef) boxes [(cling::StoredValueRef) boxes [(int *) 0x12]]
V // CHECK: (cling::StoredValueRef) boxes [(int *) 0x12]

// Savannah #96277
gCling->evaluate("gCling->declare(\"double sin(double);\"); double one = sin(3.141/2);", V);
V // CHECK: (cling::StoredValueRef) boxes [(double) 1.000000e+00]

gCling->process("double one = sin(3.141/2);", &V);
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

// Check lifetime of objects in StoredValue
.rawInput 1
struct WithDtor {
   static int fgCount;
   WithDtor() { ++fgCount; }
   WithDtor(const WithDtor&) { ++fgCount; }
   ~WithDtor() { --fgCount; }
};
int WithDtor::fgCount = 0;
WithDtor getWithDtor() { return WithDtor(); }
#include <vector>
std::vector<WithDtor> getWithDtorVec() { std::vector<WithDtor> ret; ret.resize(7); return ret; }
.rawInput 0

cling::StoredValueRef* VOnHeap = new cling::StoredValueRef();
gCling->evaluate("getWithDtor()", *VOnHeap);
*VOnHeap //CHECK: (cling::StoredValueRef) boxes [(WithDtor) @0x{{.*}}]
WithDtor::fgCount //CHECK: (int) 1
delete VOnHeap;
WithDtor::fgCount //CHECK: (int) 0

// Check destructor call for templates
VOnHeap = new cling::StoredValueRef();
gCling->evaluate("getWithDtorVec()", *VOnHeap);
*VOnHeap //CHECK: (cling::StoredValueRef) boxes [(std::vector<WithDtor>) @0x{{.*}}]
WithDtor::fgCount //CHECK: (int) 7
delete VOnHeap;
WithDtor::fgCount //CHECK: (int) 0
