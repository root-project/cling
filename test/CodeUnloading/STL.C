//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// Test stlUnloading

#include <string>
.undo
#include <string>
.undo

#include <map>
.undo
#include <map>
.undo

#include <new>
#include <string>
#include <map>

typedef std::map<int, std::string> strmap;
.undo // typedef above
.undo // #include <map>

typedef std::string teststring;
.undo // typedef above
.undo // #include <string>

#include <map>
#include <string>
typedef std::map<int, std::string> strmap;

strmap test;
test[1] = "TEST";

(const char*) test[1].c_str()
// CHECK: (const char *) "TEST"

// There are still quite a few built-ins added which make compareState
// useless. But the fact we made it to here without an error ain't bad.

// expected-no-diagnostics
.q
