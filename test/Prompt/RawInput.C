// RUN: cat %s | %cling | FileCheck %s

.rawInput 1

extern "C" int printf(const char* fmt, ...);

namespace MyNamespace {
  class MyClass {
  public:
    MyClass() { printf("MyClass constructor called!\n"); }
  };

  void f() {
    printf("Function f in namespace MyNamespace called!\n");
  }
}

.rawInput 0

MyNamespace::MyClass m = MyNamespace::MyClass(); // CHECK MyClass constructor called!
MyNamespace::f(); // CHECK: Function f in namespace MyNamespace called!
