//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.

//------------------------------------------------------------------------------
// std::source_location is a C++20 feature.
// RUN: cat %s | %cling -std=c++20 | FileCheck %s

.rawInput 1
#include <iostream>
#if __cplusplus >= 202002L
#include <version>
#endif

#ifdef __cpp_lib_source_location
#include <source_location>
std::source_location getsrcloc() {
#line 42 "CHECK_SRCLOC"
  return std::source_location::current();
}
#else
// Hack to prevent failure if __cpp_lib_source_location feature does not exist!
void getsrcloc() {
  std::cout << "(std::source_location) ";
  std::cout << "CHECK_SRCLOC:42:std::source_location getsrcloc()\n";
}
#endif
.rawInput 0

getsrcloc()
// CHECK: (std::source_location)
// CHECK: CHECK_SRCLOC:42:std::source_location getsrcloc()
