//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// XFAIL: *

// Test the ability of including a wrong file see diagnostics and remove the
// cached files so that all the changes are going to be seen next time it gets
// included.

.storeState "testUncacheFile"

#include <iostream>
#include <fstream>

// cleanup
remove("TmpClassDef.h");

// clang caches the missed too. If the file is missing it doesn't matter whether
// we create it later or not.
#include "TmpClassDef.h"

std::ofstream myfile;
myfile.open("TmpClassDef.h");
myfile << "class MyClass{};\n"
myfile << "error_here;";
myfile << "// expected-error {{C++ requires a type specifier for all declarations}}\n"
myfile.close();
#include "TmpClassDef.h"

myfile.open("TmpClassDef.h");
myfile << "class MyClass{ \n";
myfile << "public: \n";
myfile << "  int gimme12(){\n";
myfile << "    return 12;\n"
myfile << "  }\n"
myfile << "};\n";
myfile.close();
#include "TmpClassDef.h"

MyClass my;
my.gimme12()
// CHECK: (int const) 12

.compareState "testUncacheFile"
// CHECK-NOT: File with AST differencies stored in: testUncacheFileAST.diff
.q
