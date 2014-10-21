// RUN: rm -f CompGen.h.pch
// RUN: clang -x c++-header -fexceptions -fcxx-exceptions -std=c++11 %S/Inputs/CompGen.h -o CompGen.h.pch
// RUN: cat %s | %cling -I%p -Xclang -include-pch -Xclang CompGen.h.pch  2>&1 | FileCheck %s

//.storeState "a"
.x TriggerCompGen.h
.x TriggerCompGen.h
 // CHECK: I was executed
 // CHECK: I was executed
 //.compareState "a"
