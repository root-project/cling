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

class TestDecl { public: int methodDefLater(); } t
// CHECK: (class TestDecl &) @0x{{.*}}
int TestDecl::methodDefLater() { return 2; }
t.methodDefLater()
// CHECK: (int) 2

class TestDecl2 { public: float methodDefLater(); } b;
float TestDecl2::methodDefLater() { return 5.f; }
b.methodDefLater()
// CHECK: (float) 5.00000f

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


unsigned uFun(int a) {
  printf("uFun(%d)\n", a);
  return 7;
}

unsigned int uiFun(int a, int b) {
  printf("uiFun(%d, %d)\n", a, b);
  return 9;
}

uFun(6)
// CHECK: uFun(6)
// CHECK: (unsigned int) 7

uiFun(7, 8)
// CHECK: uiFun(7, 8)
// CHECK: (unsigned int) 9


static unsigned suFun(int a) {
  printf("suFun(%d)\n", a);
  return 11;
}

static unsigned int suiFun(int a, int b) {
  printf("suiFun(%d, %d)\n", a, b);
  return 13;
}

suFun(10)
// CHECK: suFun(10)
// CHECK: (unsigned int) 11

suiFun(11, 12)
// CHECK: suiFun(11, 12)
// CHECK: (unsigned int) 13

  
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

class Test2 {
  int A, B;
public:
  Test2(int a, int b);
  int subtract();
  int addition() const;
  void argspacing(int a, int b, int c) const;

  class Nested {
    public:
      struct Value {
        typedef int type;
      };
      Value::type m_A;
      Nested(int A);
      ~Nested();
      int simpleAdd(int b) const;
      int* pointer() const;
      const int& reference() const;
  };
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

Test2::Nested::Nested(int A) : m_A(A) {
  printf("Nested::Nested(%d)\n", m_A*2);
}

Test2::Nested::~Nested() {
  printf("Nested::~Nested(%d)\n", m_A);
}

Test2::Nested::Value::type Test2::Nested::simpleAdd(int b) const {
  return m_A + b;
}

Test2::Nested::Value::type* Test2::Nested::pointer() const {
  return (int*)&m_A;
}
const Test2::Nested::Value::type & Test2::Nested::reference() const {
  return m_A;
}

{
  Test2::Nested Nest(45);
  // CHECK: Nested::Nested(90)
}
// CHECK: Nested::~Nested(45)

Test2::Nested Nest2(80);
// CHECK: Nested::Nested(160)
Nest2.simpleAdd(3)
// CHECK: (int) 83

class Test2 classReturn() { return Test2(10, 11); }
classReturn()
// CHECK: Test2::Test2(10,11)
// CHECK: (class Test2) @0x{{.*}}

class Test2* classReturnPtr() { return nullptr; }
classReturnPtr()
// CHECK: (class Test2 *) nullptr

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
// CHECK: (int) 801

int * & cIntStarRef(int*& val) {
  return val;
}

int * RPtr = &Ref;
int *& RefRPtr = RPtr;
cIntStarRef(RefRPtr)
// CHECK: (int *) 0x{{[0-9a-f]+}}

namespace Issue_113 {}
// Keep the blank space after the using clause.
using namespace Issue_113; 

// FIXME: Cannot handle `X<int> func()` yet?!
template <
  class T> class X {};
namespace N { template <class T> using X = ::X<T>; }
N::X<int> funcReturnsXint() {
  return X<int>{};
}
funcReturnsXint()
// CHECK-NEXT: (N::X<int>) @0x{{.*}}

// CHECK-NEXT: Nested::~Nested(80)

// expected-no-diagnostics
