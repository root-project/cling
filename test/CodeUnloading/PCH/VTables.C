// RUN: %mkdir "%T/Rel/Path" || true
// RUN: %rm "CompGen.h.pch" && %rm "%T/Rel/Path/Relative.pch"
// RUN: clang -x c++-header -fexceptions -fcxx-exceptions -std=c++14 -pthread %S/Inputs/CompGen.h -o CompGen.h.pch
// RUN: clang -x c++-header -fexceptions -fcxx-exceptions -std=c++14 -pthread %S/Inputs/CompGen.h -o %T/Rel/Path/Relative.pch
// RUN: cat %s | %cling -I%p -Xclang -include-pch -Xclang CompGen.h.pch  2>&1 | FileCheck %s
// RUN: cat %s | %cling -I%p -I%T/Rel/Path -include-pch Relative.pch 2>&1 | FileCheck %s

//.storeState "a"
.x TriggerCompGen.h
.x TriggerCompGen.h
 // CHECK: I was executed
 // CHECK: I was executed
 //.compareState "a"
