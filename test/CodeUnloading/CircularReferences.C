// RUN: cat %s | %cling 2>&1 | FileCheck %s

.storeState "preUnload"
.rawInput 1
int g(); int f(int i) { if (i != 1) return g(); return 0; } int g() { return f(1); } int x = f(0);
.rawInput 0
.U
.compareState "preUnload"
//CHECK-NOT: Differences
float f = 1.1
//CHECK: (float) 1.1
int g = 42
//CHECK: (int) 42
.q
