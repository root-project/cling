//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S -Xclang -verify 2>&1 | FileCheck %s
// Test templated-default-args

#include <iterator>
#include <vector>
.undo
#include <vector>

std::vector<int> Vec(6, 6)
// CHECK: (std::vector<int> &) { 6, 6, 6, 6, 6, 6 }

// expected-no-diagnostics
.q
