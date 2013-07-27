// RUN: cat %s | %cling -Xclang -verify
//This file checks a pointer load operation for null prt dereference.
int *p;
int x;
x = *p; // expected-warning {{you are about to dereference null ptr, which probably will lead to seg violation. Do you want to proceed?[y/n]}}
n
class MyClass {
public:
  int a;
};
MyClass *m = 0;
if (m->a) {  printf("MyClass's a=%d", m->a);} // expected-warning {{you are about to dereference null ptr, which probably will lead to seg violation. Do you want to proceed?[y/n]}}
n
.q
