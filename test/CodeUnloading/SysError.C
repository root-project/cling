//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// Primarily for Windows
// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// Test system_error-undo

#include <system_error>
.undo
#include <system_error>
.undo

#include <string>
std::string("TEST")
// CHECK: (std::string) "TEST"

// expected-no-diagnostics
.q
