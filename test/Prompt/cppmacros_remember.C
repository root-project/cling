//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s

#include <cstdlib>
extern "C" int printf(const char* fmt, ...);
#define MYMACRO(v) if (v) { printf("string:%%s\n", v);}
#undef MYMACRO
int MYMACRO = 42; // expected-warning {{expression result unused}}
printf("MYMACRO=%d\n", MYMACRO); // CHECK: MYMACRO=42
.q
