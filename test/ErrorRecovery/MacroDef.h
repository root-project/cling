//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef HEADER_FILE_PROTECTOR
#define HEADER_FILE_PROTECTOR
int f() {
  return NN+1;
}
int n = f();
error_here; // expected-error {{use of undeclared identifier}}
#endif // HEADER_FILE_PROTECTOR
