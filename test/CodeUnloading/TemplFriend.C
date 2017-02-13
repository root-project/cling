//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S -Xclang -verify 2>&1 | FileCheck %s
// Test templated-friends

#include "TemplFriend.h"
.undo
#include "TemplFriend.h"

Test(5).Vec
// CHECK: (std::vector<int>) { 5, 5, 5, 5, 5 }

// expected-no-diagnostics
.q
