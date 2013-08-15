// RUN: cat %s | %cling -Xclang -verify | FileCheck %s
//This file checks a call instruction. The called function has arguments with nonnull attribute.
// XFAIL: *
#include <string.h>
char *p = 0;
//strcmp("a", p); // expected-warning {{you are about to dereference null ptr, which probably will lead to seg violation. Do you want to proceed?[y/n]}}

//strcmp(p, "a"); // expected-warning {{you are about to dereference null ptr, which probably will lead to seg violation. Do you want to proceed?[y/n]}}

extern "C" int printf(const char* fmt, ...);
.rawInput 1
extern "C" int cannotCallWithNull(int* p = 0);
extern "C" int cannotCallWithNull(int* p) __attribute__((nonnull(1)));
extern "C" int cannotCallWithNull(int* p);
extern "C" int cannotCallWithNull(int* p);
.rawInput 0

extern "C" int cannotCallWithNull(int* p) {
  if (!p)
    printf("Must not be called with p=0.\n");
  return 1;
}

cannotCallWithNull() // warning-expected {{null passed to a callee which requires a non-null argument}}
//CHECK-NOT: Must not be called with p=0.
//CHECK: (int) 1



.q
