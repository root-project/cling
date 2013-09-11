// RUN: cat %s | %cling -Xclang -verify
// This test verifies that we get nice warning if a method on null ptr object is
// called.
// XFAIL:*
extern "C" int printf(const char* fmt, ...);
class MyClass {
private:
  int a;
public:
  MyClass() : a(1){}
  int getA(){return a;}
};
MyClass* my = 0;
my->getA() // expected-warning {{you are about to dereference null ptr, which probably will lead to seg violation. Do you want to proceed?[y/n]}}

struct AggregatedNull {
  MyClass* m;
  AggregatedNull() : m(0) {}
}

AggregatedNull agrNull;
agrNull.m->getA(); // expected-warning {{you are about to dereference null ptr, which probably will lead to seg violation. Do you want to proceed?[y/n]}}

.q
