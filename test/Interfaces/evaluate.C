// RUN: cat %s | %cling | FileCheck %s
// XFAIL: i686, vg_leak 
// Expected to fail on 32 bit machine because we need to pass the storage object
// in a proper way for 32 bit machines. And it has invalid mem accesses.


#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"

cling::Value V;
V // CHECK: (cling::Value) <<<invalid>>> @0x{{.*}}

gCling->evaluate("1;", &V);
V // CHECK: (cling::Value) boxes [(int) 1]

long LongV = 17;
gCling->evaluate("LongV;", &V);
V // CHECK: (cling::Value) boxes [(long) 17]

int* IntP = (int*)0x12;
gCling->evaluate("IntP;", &V);
V // CHECK: (cling::Value) boxes [(int *) 0x12]

cling::Value Result;
Result.value = llvm::PTOGV(&V); // SRet
gCling->evaluate("V", &Result);
Result // CHECK: (cling::Value) boxes [(cling::Value) ]
V // CHECK: (cling::Value) boxes [(int *) 0x12]

// Savannah #96277
gCling->evaluate("double sin(double); double one = sin(3.141/2);", &V);
one // CHECK: (double) 1.000000e+00
