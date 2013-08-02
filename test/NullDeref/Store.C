// RUN: cat %s | %cling -Xclang -verify

//This file checks a pointer store operation for null ptr dereference.
int *p;
*p = 6; // expected-warning {{you are about to dereference null ptr, which probably will lead to seg violation. Do you want to proceed?[y/n]}}
.q
