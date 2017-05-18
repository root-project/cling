//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify

class MyClass{}; // expected-note {{previous definition is here}}
struct MyClass{} // expected-error {{redefinition of 'MyClass'}}
MyClass * s; // expected-note {{previous definition is here}}
MyClass s; // expected-error {{redefinition of 's'}}

const char* a = "test"; // expected-note {{previous definition is here}}
const char* a = ""; // expected-error {{redefinition of 'a'}}

.q
