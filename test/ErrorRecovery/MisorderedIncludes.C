//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s
// Test misorderedIncludes


#include <map>
.undo
#include <map>
.undo

#include <map>
std::map<int, std::string> test;
#include <string>
std::map<int, std::string> test;

test[0] = "A";
(const char*) test[0].c_str()
// CHECK: (const char *) "A"

test[10] = "B";
(const char*) test[10].c_str()
// CHECK: (const char *) "B"

test[20] = "C";
(const char*) test[20].c_str()
// CHECK: (const char *) "C"

test[30] = "LONGER";
(const char*) test[30].c_str()
// CHECK: (const char *) "LONGER"

.q
