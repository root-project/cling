// RUN: cat %s | %cling -Xclang -verify
//This file checks a pointer load operation for null prt dereference.
// XFAIL: i686-pc-linux-gnu
int *p = 0;;
double x;
x = double(*p); // expected-warning {{null passed to a callee which requires a non-null argument}}

void *q = 0;
int y;
y = int(*(int *)q); // expected-warning {{null passed to a callee which requires a non-null argument}}
