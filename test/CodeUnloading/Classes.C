// RUN: cat %s | %cling 2>&1 | FileCheck %s

extern "C" int printf(const char* fmt, ...);
.storeState "preUnload"
class MyClass{
private:
  double member;
public:
  MyClass() : member(42){}
  static int get12(){ return 12; } 
  double getMember(){ return member; } 
}; MyClass m; m.getMember(); MyClass::get12();
.U
.compareState "preUnload"
//CHECK-NOT: Differences
float MyClass = 1.1
//CHECK: (float) 1.1
.q
