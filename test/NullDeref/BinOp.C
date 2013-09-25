// RUN: cat %s | %cling -Xclang -verify
//This file checks a pointer load operation for null prt dereference.
// XFAIL: i686-pc-linux-gnu
int *p = 0;
int x;
x = *p + 2; // expected-warning {{null passed to a callee which requires a non-null argument}}
x = 2 + *p; // expected-warning {{null passed to a callee which requires a non-null argument}}

x = *p > 2; // expected-warning {{null passed to a callee which requires a non-null argument}}
x = 2 > *p; // expected-warning {{null passed to a callee which requires a non-null argument}}
