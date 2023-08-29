//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s

#include <memory>
#include <string>

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
.undo
.compareState "preUnload"
//CHECK-NOT: Differences
float MyClass = 1.1
//CHECK: (float) 1.10000f

template <typename T>
struct MyStruct { T f(T x) { return x; } };
MyStruct<float> obj;
obj.f(42.0)
//CHECK: (float) 42.0000f
.undo
obj.f(42.0)
//CHECK: (float) 42.0000f

auto p = std::make_unique<std::string>("string");
(unsigned long)p.size() // expected-error{{no member named 'size' in 'std::unique_ptr<std::basic_string<char>>'; did you mean to use '->' instead of '.'?}}
(unsigned long)p->size()
//CHECK: (unsigned long) 6

.q
