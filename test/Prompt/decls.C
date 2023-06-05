//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p | FileCheck %s
#include <cmath>
#include <iostream>

struct S{int i;} ss;
S s = {12 };

struct U{void f() const {};} uu;

struct V{V(): v(12) {}; int v; } vv;

struct typeA {
  typeA(int i): num{i} {};
  typeA();
  ~typeA();

  typeA& operator()(int k);
  const typeA& operator[](int k) const;

  operator int() const;

  int num;
};

typeA::typeA() = default;
::typeA::~typeA() {};
::typeA& ::typeA::operator()(int k) { return *this; }
const typeA& typeA::operator[](int k) const { return *this; }
bool operator>=(const typeA &lhs, int t) { return lhs.num >= t; }
bool operator<=(::typeA const &lhs, int t) { return lhs.num <= t; }
bool operator<(typeA const&lhs, int t) { return lhs.num < t; }
bool operator>(const typeA &lhs, int t) { return lhs.num > t; }
::typeA operator+(const typeA &lhs, const ::typeA &rhs) {
  return typeA{lhs.num + rhs.num};
}
::typeA operator "" _tA(unsigned long long int t) { return typeA{static_cast<int>(t)}; }
std::ostream& operator<<(std::ostream& os, const typeA& a) {return (os << a.num);}
typeA::operator int() const { return num + 7; }

std::cout << 76601_tA; // CHECK:76601
6551_tA(99)[4].num // CHECK:6551
typeA{-675} > 0 // CHECK:false
99_tA >= 99 // CHECK:true
::typeA(31) < 31 // CHECK:false
(60_tA + 3_tA).num // CHECK:63
static_cast<int>(18_tA)  // CHECK:25

::atoi("42") // CHECK:42

int i = 12;
float f = sin(12);
int j = i;
extern "C" int printf(const char* fmt, ...);
printf("j=%d\n",j); // CHECK:j=12
#include <string>
std::string str("abc");
printf("str=%s\n",str.c_str()); // CHECK: str=abc

[[nodiscard]] int f() { return 0; }
void g() { f(); } // expected-warning@1 {{ignoring return value of function declared with 'nodiscard' attribute}}
// -Wunused-result is filtered for code parsed via `Interpreter::EvaluateInternal()`
f();

.q
