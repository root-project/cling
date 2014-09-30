// RUN: rm -f Inlines.h.pch
// RUN: clang -cc1 -target-cpu x86-64 -fdeprecated-macro -fmath-errno -fexceptions -fcxx-exceptions -fgnu-keywords -emit-pch -xc++ -std=c++11 %S/Inputs/Inlines.h -o Inlines.h.pch
// RUN: cat %s | %cling -I%p -Xclang -trigraphs -Xclang -include-pch -Xclang Inlines.h.pch  2>&1 | FileCheck %s

//XFAIL:*
#include "Inputs/Inlines.h"
extern "C" int printf(const char*, ...);
CompGen a;
CompGen b = a;
//.storeState "a"
printf("%d %d\n", a.InlineFunc(), b.InlineFunc());
.undo 1
printf("%d %d\n", a.InlineFunc(), b.InlineFunc());
 //.compareState "a"

 // CHECK: I was executed
 // CHECK: I was executed
.q
