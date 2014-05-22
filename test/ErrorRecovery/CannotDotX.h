//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

extern "C" int printf(const char* fmt, ...);

struct MyClass {
  MyClass() { printf("MyClass ctor called!\n"); }
  ~MyClass() { printf("MyClass dtor called!\n"); }
};
