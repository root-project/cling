// RUN: cat %s | %cling | FileCheck %s

#include <cstdlib>
extern "C" int printf(const char* fmt, ...);
#define MYMACRO(v) if (v) { printf("string:%%s\n", v);}
#undef MYMACRO
int MYMACRO = 42; // expected-warning {{expression result unused}}
printf("MYMACRO=%d\n", MYMACRO); // CHECK: MYMACRO=42
.q
