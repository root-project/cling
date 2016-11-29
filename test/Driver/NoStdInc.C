//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// This test is dependent on IncrementalParser::Initialize
// calling ParseInternal("#include <new>");
// and the structure of the warning message 
// It will also fail when it is not possible to verify matching C++ ABIs

// %nostdincxx is -nostdinc on Windows, -nostdinc++ everywhere else.
// RUN: cat %s | %cling %nostdincxx -Xclang -verify 2>&1 | FileCheck %s
// Test nobuiltinincTest

// expected-error@1 {{'new' file not found}}

// CHECK: Warning in cling::IncrementalParser::CheckABICompatibility():
// CHECK:  Possible C++ standard library mismatch, compiled with {{.*$}}

.q
