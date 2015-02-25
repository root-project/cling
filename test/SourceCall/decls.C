//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling %s | FileCheck %s
#include <cmath>
#include <stdio.h>

struct S{int i;};
S s = {12 };

typedef struct {int i;} T;

struct U{void f() const {};};

struct V{V(): v(12) {}; int v; };

int i = 12;
float f = sin(12);
int j = i;

void decls() {
   int arg1 = 17, arg2 = 42, add = -1;
#ifdef __linux__
   __asm__ ( "addl %%ebx, %%eax;" : "=a" (add) : "a" (arg1) , "b" (arg2) );
#else
   add = arg1 + arg2;
#endif
   printf("result=%d\n", add); // CHECK:result=59
   printf("j=%d\n",j); // CHECK:j=12
}
