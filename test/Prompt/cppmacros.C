//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling %s | FileCheck %s
extern "C" int printf(const char* fmt, ...);
#define MYMACRO(v) \
   if (v) { \
      printf("string:%s\n", v);\
   }

void cppmacros() {
   MYMACRO("PARAM"); // CHECK: string:PARAM
}

#pragma clang diagnostic ignored "-Wkeyword-compat" // ROOT-6531

#pragma once // ROOT-7113
