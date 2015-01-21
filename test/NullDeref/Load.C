//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify
//This file checks a pointer load operation for null prt dereference.
int *p;
int x;
x = *p; // expected-warning {{null passed to a callee that requires a non-null argument}}

extern "C" int printf(const char* fmt, ...);
class MyClass {
public:
  int a;
};
MyClass *m = 0;
if (m->a) {  printf("MyClass's a=%d", m->a);} // expected-warning {{null passed to a callee that requires a non-null argument}}
.q
