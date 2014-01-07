//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify

class MyClass{};
class MyClass{} // expected-error {{redefinition of 'MyClass'}} expected-note {{previous definition is here}}
MyClass s;
MyClass s; // expected-error {{redefinition of 's'}} expected-note {{previous definition is here}}

const char* a = "test";
const char* a = ""; // expected-error {{redefinition of 'a'}} expected-note {{previous definition is here}}

.q
