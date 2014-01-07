//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// Makes it easy to debug
typedef int A;
typedef int B;
typedef int C;

extern A my_int;
extern B my_int;
extern C my_int;

int my_int = 10;

extern A my_funct();
extern B my_funct();

int my_funct() {
  return 10;
}

error_here; // expected-error {{C++ requires a type specifier for all declarations}}
