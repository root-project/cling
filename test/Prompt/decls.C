// RUN: cat %s | %cling -I%p | FileCheck %s
#include <cmath>

struct S{int i;} ss;
S s = {12 };

struct U{void f() const {};} uu;

struct V{V(): v(12) {}; int v; } vv;

int i = 12;
float f = sin(12);
int j = i;
extern "C" int printf(const char* fmt, ...);
printf("j=%d\n",j); // CHECK:j=12
#include <string>
std::string str("abc");
printf("str=%s\n",str.c_str()); // CHECK: str=abc

.q
