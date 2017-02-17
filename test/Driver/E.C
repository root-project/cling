//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: echo "" | %cling -dM -E -xc++ -std=c++11 | FileCheck %s

// CHECK: #define __CLING__ 1
// CHECK: #define __CLING__CXX11 1
// CHECK: #define __CLING__{{GNUC|clang|MSVC}}

.q
