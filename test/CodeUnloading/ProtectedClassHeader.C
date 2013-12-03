// RUN: cat %s | %cling -I%p
.storeState "a"
#include "ProtectedClass.h"
.U
.compareState "a"
// CHECK-NOT: Differences
