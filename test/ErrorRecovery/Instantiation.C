//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S -Xclang -verify 2>&1 | FileCheck %s
// Test instantiationRecover

extern "C" int printf(const char*,...);

template <class T> static void bracketIt(const T& iter, const char *f = " %d") {
  printf("{");
  for (auto val : iter )
    printf(f, val);
  printf(" }\n");
}

#define TEST_TMPLT
#define TEST_CLASS TestV
#define TEST_VALT  int
#include "Instantiation.h"
TestV tv;
for (auto val : tv(10) ) { *val; }
// expected-error@2 {{indirection requires pointer operand ('int' invalid)}}
bracketIt(tv(10));
// CHECK: { 0 1 2 3 4 5 6 7 8 9 }
#undef TEST_TMPLT
#undef TEST_CLASS
#undef TEST_VALT


#define TEST_TMPLT template <class T>
#define TEST_CLASS TestT
#define TEST_VALT  T
#include "Instantiation.h"
TestT<float> tt;
for (auto val : tt(15) ) { val->failure; }
// expected-error@2 {{member reference type 'float' is not a pointer}}
bracketIt(tt(15), " %.1f");
// CHECK: { 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 }
#undef TEST_TMPLT
#undef TEST_CLASS
#undef TEST_VALT


#include "Instantiation.h"

TestStatic<int> ts0;
for (auto val : ts0 ) { val(); }
// expected-error@2 {{called object type 'int' is not a function or function pointer}}
bracketIt(ts0);
// CHECK: { 0 }

TestStatic<int> ts1;
for (auto val : ts1 ) { val(); }
// expected-error@2 {{called object type 'int' is not a function or function pointer}}
bracketIt(ts1);
// CHECK: { 0 1 }

TestStatic<int> ts2;
for (auto val : ts2 ) { val(); }
// expected-error@2 {{called object type 'int' is not a function or function pointer}}
bracketIt(ts2);
// CHECK: { 0 1 2 }


TestIterInst<int> ti;
for (auto val : ti.test1(12) ) { val(); }
// expected-error@2 {{called object type 'int' is not a function or function pointer}}
bracketIt(ti.test1(12));
// CHECK: { 0 1 2 3 4 5 6 7 8 9 10 11 }

TestIterInst<int> ti2;
for (auto val : ti2.test2(20) ) { val.failure; }
// expected-error@2 {{member reference base type 'int' is not a structure or union}}
bracketIt(ti2.test2(20));
// CHECK: { 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 }

ti2.access()
// CHECK: (bool) true



#include <vector>
std::vector<int> v = { 0, 1, 2, 3, 4, 5 };
for (auto val : v ) { val[5]; }
// expected-error@2 {{subscripted value is not an array, pointer, or vector}}
bracketIt(v);
// CHECK: { 0 1 2 3 4 5 }


#include <string>
std::string test("later");
(const char*) test.c_str()
// CHECK: (const char *) "later"



TestStatic<int> ts3;
for (auto val : ts2 ) { val(); }
// expected-error@2 {{called object type 'int' is not a function or function pointer}}
bracketIt(ts3);
// CHECK: { 0 1 2 3 }


.q
