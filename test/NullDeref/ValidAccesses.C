//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify
// This test verifies that we do not produce a warning when
// - an iterator is derefed;
// - a lambda function is derefed.

#include <vector>

class MyClass;

std::vector<MyClass*> vect(3);
for (auto it = vect.begin(); it != vect.end(); ++it) MyClass* ptr = *it;
 // expected-no-diagnostics

auto lamb =[](int x){return x;};
*lamb;
 // expected-no-diagnostics

.q

