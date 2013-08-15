// RUN: cat %s | %cling -Xclang -verify
//This file checks a call instruction. The called function has arguments with nonnull attribute.
#include <string.h>
char *p;
strcmp("a", p); // expected-warning {{you are about to dereference null ptr, which probably will lead to seg violation. Do you want to proceed?[y/n]}}

strcmp(p, "a"); // expected-warning {{you are about to dereference null ptr, which probably will lead to seg violation. Do you want to proceed?[y/n]}}
.q
