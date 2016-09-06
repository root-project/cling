//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify

struct {int j;}; // expected-error {{anonymous structs and classes must be class members}}

// ROOT-7610
do { int a = 0; } while(a==0); // expected-error {{use of undeclared identifier 'a'}}
.q
