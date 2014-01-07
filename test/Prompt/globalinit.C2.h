//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

extern "C" int printf(const char*,...);

struct B {
   struct S {
      S() { printf("B::S()\n"); }
      ~S() { printf("B::~S()\n"); }
   };
   static S s;
};
B::S B::s;
