// RUN: cat %s | %cling | FileCheck %s

// XFAIL:*
// The test exposes a weakness in the declaration extraction of types. As 
// reported in issue ROOT-5248.

extern "C" int printf(const char* fmt, ...);

class MyClass;
extern MyClass* my;
class MyClass {
public: MyClass* getMyClass() {
  printf("Works!\n");
  return 0;
}
} cl;
MyClass* my = cl.getMyClass();
.q
//CHECK: Works!
