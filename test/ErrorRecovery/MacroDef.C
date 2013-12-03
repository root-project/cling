// RUN: cat %s | %cling -I%p -Xclang -verify 2>&1

.rawInput 1
extern "C" int printf(const char* fmt, ...);
#define NN 5
int printNN() {
  printf("NN=%d", NN);
  return 0;
}
.rawInput 0
printNN();
.storeState "MacroDef"
#include "MacroDef.h"
.compareState "MacroDef"
// CHECK-NOT: Differences
printNN();
