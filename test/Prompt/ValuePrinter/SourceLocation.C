//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.

//------------------------------------------------------------------------------
// RUN: cat %s | %cling | FileCheck %s

#include <iostream>
#if __cplusplus >= 202002L
#include <version>
#endif

#ifndef __cpp_lib_source_location
// Hack to prevent failure if __cpp_lib_source_location feature does not exist!
std::cout << "(std::source_location) ";
std::cout << "CHECK_SRCLOC:42:std::source_location getsrcloc()\n";
#else
#include <source_location>
std::source_location getsrcloc() {
#line 42 "CHECK_SRCLOC"
  return std::source_location::current();
}
getsrcloc()
#endif
// CHECK: (std::source_location)
// CHECK: CHECK_SRCLOC:42:std::source_location getsrcloc()
