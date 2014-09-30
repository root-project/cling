// RUN: rm -f CompGen.h.pch
// RUN: clang -cc1 -target-cpu x86-64 -fdeprecated-macro -fmath-errno -fexceptions -fcxx-exceptions -fgnu-keywords -emit-pch -xc++ -std=c++11 %S/CompGen.h -o CompGen.h.pch
// RUN: cat %s | %cling -I%p -Xclang -trigraphs -Xclang -include-pch -Xclang CompGen.h.pch  2>&1 | FileCheck %s

//.storeState "a"
.x TriggerCompGen.h
.x TriggerCompGen.h
 // CHECK: I was executed
 // CHECK: I was executed
 //.compareState "a"
