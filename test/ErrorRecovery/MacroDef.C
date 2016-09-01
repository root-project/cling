//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p -Xclang -verify 2>&1

extern "C" int printf(const char* fmt, ...);
#define NN 5
int printNN() {
  printf("NN=%d", NN);
  return 0;
}

printNN();
.storeState "MacroDef"
#include "MacroDef.h"
.compareState "MacroDef"
// CHECK-NOT: Differences
printNN();
