//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling -C -E -P  %s | %cling -nostdinc++ -Xclang -verify 2>&1 | FileCheck %s
// RUN: %cling -C -E -P -DCLING_NO_BUILTIN %s | %cling -nostdinc++ -nobuiltininc -Xclang -verify 2>&1 | FileCheck %s

// expected-error@input_line_1:1 {{'new' file not found}}

//      CHECK: Warning in cling::IncrementalParser::CheckABICompatibility():
// CHECK-NEXT:  Failed to extract C++ standard library version.

.q
