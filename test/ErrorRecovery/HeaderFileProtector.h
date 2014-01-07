//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef HEADER_FILE_PROTECTOR
#define HEADER_FILE_PROTECTOR
#define NN 5
#undef NN
#define NN 6
#define P 5

class MyHeaderFileProtectedClass {};
#undef P
error_here; // expected-error {{C++ requires a type specifier for all declarations}}
#endif // HEADER_FILE_PROTECTOR
