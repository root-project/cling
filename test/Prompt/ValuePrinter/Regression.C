//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s

// This file should be used as regression test for the value printing subsystem
// Reproducers of fixed bugs should be put here

// PR #93006
.rawInput 1
extern "C" int printf(const char* fmt, ...);
class A {
public:
  int Var;
  A(int arg) { Var = arg; }
  int someFunc(float) { return 42; }
  ~A() { printf("A d'tor\n"); }
};

const A& foo(const A& arg) { return arg; }
A foo2(const A& arg) { return A(42); }
.rawInput 0

foo(A(12)).Var
// CHECK: (const int) 12
// CHECK: A d'tor
// End PR #93006

// myvector.end() failed to print (roottest/cling/stl/default/VectorSort.C)
foo2(A(42))
// CHECK: (A) @0x{{[0-9a-f]+}}
// CHECK: A d'tor

 // Savannah #96523
int *p = (int*)0x123;
p // CHECK: (int *) 0x123
const int *q = (int*)0x123;
q // CHECK: (const int *) 0x123

0.00001234L // CHECK: (long double) 1.234e-05L

// PR ROOT-5467
&A::someFunc // CHECK: (int (A::*)(float)) Function @0x{{[0-9a-f]+}}

nullptr // CHECK: (nullptr_t) nullptr

true // CHECK: (bool) true
false // CHECK: (bool) false

#include <unordered_set>
std::unordered_multiset<float> {1} // ROOT-7310

// ROOT-7426
#include <string>
#include <vector>
std::vector<std::string> a = { "a", "b", "c" } // CHECK: (std::vector<std::string> &) { "a", "b", "c" }
a // CHECK: (std::vector<std::string> &) { "a", "b", "c" }
a[0] // CHECK: ({{.*}} &) "a"

// ROOT-7918
enum H {  h = (unsigned long long )-1 };
h // CHECK: (H) (H::h) : (unsigned long{{( long)?}}) 18446744073709551615

// ROOT-7837
auto bla=[](double *x, double *par, int blub){return x[0]*blub;} // CHECK: ((lambda) &) @0x

#include <functional>
using namespace std::placeholders;
auto fn_moo = std::bind (bla, _1,_2,10) // CHECK: ERROR in cling::executePrintValue(): missing value string.
// Make sure cling survives
12 // CHECK: (int) 12

// ROOT-8077
.rawInput 1
#include <string>
void f(std::string) {}
.rawInput 0
f // CHECK: (void (*)(std::string)) Function @0x{{[0-9a-f]+}}
// CHECK: at :1:
// CHECK: void f(std::string) {}
