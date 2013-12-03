// RUN: cat %s | %cling -I%p -Xclang -verify 2>&1
.storeState "a"
#include "HeaderFileProtector.h"
.compareState "a"
// CHECK-NOT: Differences
