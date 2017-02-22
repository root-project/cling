// RUN: %rm "CompGen2.h.pch"
// RUN: %cling -x c++-header %S/Inputs/CompGen.h -o CompGen2.h.pch
// RUN: cat %s | %cling -I%p -Xclang -include-pch -Xclang CompGen2.h.pch  2>&1 | FileCheck %s

//.storeState "a"
.x TriggerCompGen.h
.x TriggerCompGen.h
 // CHECK: I was executed
 // CHECK: I was executed
 //.compareState "a"
