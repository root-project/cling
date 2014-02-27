//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s
//XFAIL:*
extern "C" int printf(const char* fmt, ...);
.storeState "preUnload"
class ClassWithDtor{
private:
  int N;
public:
  ClassWithDtor() : N(0){ N++; }
  ~ClassWithDtor() {
    N--;
    printf("Dtor called, N=%d\n", N);
  }
}; ClassWithDtor m;
.U
//CHECK: Dtor called, N=0
.compareState "preUnload"

//CHECK-NOT: Differences
.q
