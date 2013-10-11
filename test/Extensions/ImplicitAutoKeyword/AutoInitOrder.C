// RUN: cat %s | %cling --enable-implicit-auto-keyword | FileCheck %s
// XFAIL: *

class MyClass {
public:
  int i;
  MyClass(int a) : i(a) {};
};

MyClass* my1 = new MyClass(0); my1->i = 10; my2 = new MyClass(my1->i); my1->i++;
my1->i
//CHECK: (int) 11
my2->i
//CHECK: (int) 10
.q
