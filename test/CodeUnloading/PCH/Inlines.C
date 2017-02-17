// RUN: %rm "Inlines.h.pch"
// RUN: %cling -x c++-header %S/Inputs/Inlines.h -o Inlines.h.pch
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
