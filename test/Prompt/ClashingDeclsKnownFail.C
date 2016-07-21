//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify
// XFAIL: *

// The test exposes a weakness in the declaration extraction of types. As
// reported in issue ROOT-5248.

class MyClass;
extern MyClass* my;
class MyClass {
public:
  MyClass* getMyClass() {
    return 0;
  }
} cl;
// The next line should work without complaints!
MyClass* my = cl.getMyClass();
