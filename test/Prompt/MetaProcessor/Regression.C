// RUN: cat %s | %cling -I%p | FileCheck %s

// This file should be used as regression test for the meta processing subsystem
// Reproducers of fixed bugs should be put here

// PR #93092
// Don't remove the spaces and tabs
.L     cling/Interpreter/Interpreter.h    
.x ./DotXable.h(5)
// CHECK: 5
// End PR #93092
