//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling --ptrcheck -Xclang -verify
// XFAIL: powerpc64
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

class Left {
  int m_LeftValue;
public:
  Left() : m_LeftValue(-1) {}
  int getLeftValue() { return m_LeftValue; }
  void setLeftValue(int v) { m_LeftValue = v; }
};

class Right {
  int m_RightValue;
public:
  Right() : m_RightValue(-2) {}
  int getRightValue() { return m_RightValue; }
   void setRightValue(int v) { m_RightValue = v; }
};

class Bottom: public Right, public Left {
   int m_BottomValue;
public:
   Bottom() : m_BottomValue(-3) {}
   int getBottomValue() { return m_BottomValue; }
   void setBottomValue(int v) { m_BottomValue = v; }
};

template <typename T> void TemplateFunc() {
   T *b = new T;
   b->setBottomValue(3);
   b->setRightValue(2);
   b->setLeftValue(1);

   if (b->getBottomValue() != 3)
      printf("fail: expected bottom value to be 3 but got %d\n",
             b->getBottomValue());
   if (b->getRightValue() != 2)
      printf("fail: expected right value to be 3 but got %d\n",
             b->getRightValue());
   if (b->getLeftValue() != 1)
      printf("fail: expected left value to be 3 but got %d\n",
             b->getLeftValue());
}

template <typename Q> void TemplateFuncUnrelated() {
   Bottom *b = new Bottom;
   b->setBottomValue(3);
   b->setRightValue(2);
   b->setLeftValue(1);

   if (b->getBottomValue() != 3)
      printf("fail: expected bottom value to be 3 but got %d\n",
             b->getBottomValue());
   if (b->getRightValue() != 2)
      printf("fail: expected right value to be 3 but got %d\n",
             b->getRightValue());
   if (b->getLeftValue() != 1)
      printf("fail: expected left value to be 3 but got %d\n",
             b->getLeftValue());
}

TemplateFunc<Bottom>();
TemplateFuncUnrelated<MyClass>();
.q
