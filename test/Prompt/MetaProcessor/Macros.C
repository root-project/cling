// RUN: cat %s | %cling | FileCheck %s
//XFAIL: *
// This test should test the unnamed macro support once it is moved in cling.
.x Commands.macro
// CHECK: I am a function called f.
// CHECK-NOT: 0
