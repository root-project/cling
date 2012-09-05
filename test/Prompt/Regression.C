// RUN: cat %s | %cling -I%p | FileCheck %s

// This file should be used as regression test for the meta processing subsystem
// Reproducers of fixed bugs should be put here

// PR #96277
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"
#include <stdio.h>
gCling->declare("int print() { printf(\"print is run.\\n\"); return 1; }");
cling::Value V;
gCling->process("int a = print();",&V);
//CHECK: print is run.
gCling->process("a", &V);
//CHECK: (int const) 1
gCling->process("a;", &V);
//CHECK-NOT: print is run.
// End PR #96277
