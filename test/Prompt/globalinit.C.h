//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

extern "C" int printf(const char*,...);

struct A {
   struct S {
      S() { printf("A::S()\n"); }
      ~S() { printf("A::~S()\n"); }
   };
   static S s;
};
A::S A::s;
