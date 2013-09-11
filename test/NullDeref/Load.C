// RUN: cat %s | %cling -Xclang -verify
//This file checks a pointer load operation for null prt dereference.
int *p;
int x;
x = *p; // expected-warning {{null passed to a callee which requires a non-null argument}}

extern "C" int printf(const char* fmt, ...);
class MyClass {
public:
  int a;
};
MyClass *m = 0;
if (m->a) {  printf("MyClass's a=%d", m->a);} // expected-warning {{null passed to a callee which requires a non-null argument}}
.q
