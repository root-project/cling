// RUN: cat %s | %cling -Xclang -verify

//This file checks a pointer store operation for null ptr dereference.
// XFAIL: i686-pc-linux-gnu
int *p;
*p = 6; // expected-warning {{null passed to a callee which requires a non-null argument}}
.q
