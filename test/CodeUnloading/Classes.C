//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

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
.undo
.compareState "preUnload"
//CHECK-NOT: Differences
float MyClass = 1.1
//CHECK: (float) 1.1
.q
