//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s
int a = 12;
a // CHECK: (int) 12

const char* b = "b" // CHECK: (const char *) "b"
   const char* n = 0 // CHECK: (const char *) nullptr

struct C {int d;} E = {22};
E // CHECK: (struct C &) @0x{{[0-9A-Fa-f]{5,12}.}}
E.d // CHECK: (int) 22

#include <string>
std::string s("xyz")
// CHECK: (std::string &) "xyz"

#include <limits.h>
class Outer {
public:
  struct Inner {
    enum E{
      A = INT_MAX,
      B = 2,
      C = 2,
      D = INT_MIN
    } ABC;
  };
};
Outer::Inner::C
// CHECK: (Outer::Inner::B) ? (Outer::Inner::C) : (int) 2
Outer::Inner::D
// CHECK: (Outer::Inner::D) : (int) -{{[0-9].*}}

// Put an enum on the global scope
enum E{ e1 = -12, e2, e3=33, e4, e5 = 33};
e2
// CHECK: (e2) : (int) -11
::e1
// CHECK: (e1) : (int) -12


// Arrays:
float farr[] = {0.,1.,2.,3.,4.,5.} // CHECK: (float [6]) { 0.{{0+}}f, 1.{{0+}}f, 2.{{0+}}f, 3.{{0+}}f, 4.{{0+}}f, 5.{{0+}}f }
std::string sarr[3] = {"A", "B", "C"} // CHECK: (std::string [3]) { "A", "B", "C" }

typedef void (*F_t)(int);

F_t fp = 0;
fp // CHECK: (F_t) Function @0x0
#include <stdio.h>
fp = (F_t)printf // CHECK: (F_t) Function @0x{{[0-9A-Fa-f]{5,12}.}}
.q
