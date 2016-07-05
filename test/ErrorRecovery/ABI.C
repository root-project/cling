//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -nostdinc++ -nobuiltininc -Xclang -verify 2>&1 | FileCheck %s

// expected-error {{'new' file not found}}

// CHECK: Warning in cling::IncrementalParser::CheckABICompatibility():
// CHECK:  Possible C++ standard library mismatch, compiled with

.q
