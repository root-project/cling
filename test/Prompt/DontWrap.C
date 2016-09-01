//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// Test dontWrapDefinitions

extern "C" int printf(const char*,...);

static int staticFunc(int a) {
  printf("staticFunc(%d)\n", a);
  return 1;
}

static int staticFunc(int a, int b, int c) {
  printf("staticFunc(%d, %d, %d)\n", a, b, c);
  return 3;
}

staticFunc(10)
// CHECK: staticFunc(10)
// CHECK: (int) 1

staticFunc(10, 20, 30)
// CHECK: staticFunc(10, 20, 30)
// CHECK: (int) 3


int localFun(int a) {
  printf("localFun(%d)\n", a);
  return 1;
}

#define staticweird int
staticweird localFun(int a, int b, int c) {
  printf("localFun(%d, %d, %d)\n", a, b, c);
  return 3;
}

localFun(0)
// CHECK: localFun(0)
// CHECK: (int) 1

localFun(5, 6, 7)
// CHECK: localFun(5, 6, 7)
// CHECK: (int) 3
  
class Test {
public:
  Test(int a, int b);
  Test();
  ~Test();
};

Test::Test(int a, int b) {
  printf("Test::Test(%d,%d)\n", a, b);
}

Test::Test() {
  printf("Test::Test\n");
}

Test::~Test() {
  printf("Test::~Test\n");
}

{
  Test t;
  // CHECK: Test::Test
}
// CHECK: Test::~Test

{
  Test t(5, 6);
  // CHECK: Test::Test(5,6)
}
// CHECK: Test::~Test

extern "C" int printf(const char*,...);

class Test2 {
  int A, B;
public:
  Test2(int a, int b);
  int subtract();
  int addition() const;
  void argspacing(int a, int b, int c) const;

};

Test2::Test2(int a, int b) : A(a), B(b) {
  printf("Test2::Test2(%d,%d)\n", A, B);
}
int Test2::subtract() {
  return A-B;
}
int Test2::addition() const {
  return A+B;
}
void Test2::argspacing(int a,
                       int
                           b,
                       int 
                         
                         
                           c) const {
  printf("Test2::argspacing(%d,%d,%d)\n", a, b, c);
}


Test2 t0(4, 5);
// CHECK: Test2::Test2(4,5)

t0.subtract()
// CHECK: (int) -1

t0.addition()
// CHECK: (int) 9

t0.argspacing(1,2,3)
// CHECK: Test2::argspacing(1,2,3)

int Ref = 42;
const int& cIntRef(const int &val) {
  return val;
}
cIntRef(Ref)
// CHECK: (const int) 42

Ref = 32;
const int* cIntStar(const int* val) {
  return val;
}
*cIntStar(&Ref)
// CHECK: (const int) 32


Ref = 56;
int & cIntRefSpace(int &val) {
  return val;
}
cIntRefSpace(Ref)
// CHECK: (int) 56

Ref = 74;
int * cIntStarSpace(int* val) {
  return val;
}
*cIntStarSpace(&Ref)
// CHECK: (int) 74


constexpr int cExpr() {
  return 801;
}
cExpr()
// CHECK: (int) 8

int * & cIntStarRef(int*& val) {
  return val;
}

int * RPtr = &Ref;
int *& RefRPtr = RPtr;
cIntStarRef(RefRPtr)
// CHECK: (int *) 0x{{[0-9]+}}

// expected-no-diagnostics
