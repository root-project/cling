//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s
// XFAIL:*
extern "C" int printf(const char* fmt, ...);
// force emission of cxa_atexit such that it doesn't pollute the diff.
class MyClass{public: ~MyClass(){} }mm;
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
.undo
//CHECK: Dtor called, N=0
.compareState "preUnload"
//CHECK-NOT: Differences



// Make sure that the static template member inits get unloaded correctly.
// See CodeGenModule::EmitCXXGlobalVarDeclInitFunc() - they get emitted *next*
// to GLOBAL__I_a, not as call nodes within GLOBAL__I_a.
.storeState "preUnload3"

struct XYZ {
   XYZ(int I = -10): m(I) {}
   int m;
};

template <typename T> struct S {
   static XYZ one;
   static XYZ two;
};
template <typename T> XYZ S<T>::one = XYZ(12);
template <typename T> XYZ S<T>::two = XYZ(17);

XYZ a = XYZ(12);
XYZ b = XYZ(12);

int T(){
   S<int> o;
   return o.one.m;
}

.undo 7
.compareState "preUnload3"




// Make sure we have exactly one symbol of ~X(), i.e. that the unloading does
// not remove it and CodeGen re-emits in upon seeing a new use in X c;
12  // This is a temporary fix, not to allow .undo to try to unload RuntimePrintvalue.h
    // If this is not here, the test hangs on first .undo 3 below. Should be investigated further.
.storeState "preUnload2"

extern "C" int printf(const char*, ...);
struct X {
   X(): i(12) {}
   ~X() { static int I = 0; printf("~X: %d\n", ++I); }
   int i;
};
X a;
int S() {
   X b;
   return a.i + b.i;
}

S() // CHECK: (int) 24
.undo 3 // Remove up to "X a;"
// CHECK-NEXT: ~X: 1
// CHECK-NEXT: ~X: 2
X c;
.undo 3
// CHECK-NEXT: ~X: 3
.compareState "preUnload2"



.q
