//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify
// This test verifies that we get nice warning if a method on null ptr object is
// called.

extern "C" int printf(const char* fmt, ...);
class MyClass {
private:
  int a;
public:
  MyClass() : a(1){}
  int getA(){return a;}
};
MyClass* my = 0;
my->getA() // expected-warning {{null passed to a callee that requires a non-null argument}}

struct AggregatedNull {
  MyClass* m;
  AggregatedNull() : m(0) {}
}

AggregatedNull agrNull;
agrNull.m->getA(); // expected-warning {{null passed to a callee that requires a non-null argument}}

.q
