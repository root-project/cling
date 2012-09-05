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
   printf("j=%d\n",j); // CHECK:j=12
}
