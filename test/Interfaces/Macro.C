//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

#include "cling/Interpreter/Interpreter.h"

#define TEST01 "A  B  C  D  E  F"
gCling->getMacroValue("TEST01")
// CHECK: (std::string) "A  B  C  D  E  F"

#define TEST02 0  1  2  3  4  5  6  7
gCling->getMacroValue("TEST02")
// CHECK-NEXT: (std::string) "0 1 2 3 4 5 6 7"

#define TEST03 STRIP "STRING" TEST
gCling->getMacroValue("TEST03")
// CHECK-NEXT: (std::string) "STRIP STRING TEST"

#define TEST03 STRIP "STRING" TEST
gCling->getMacroValue("TEST03", 0)
// CHECK-NEXT: (std::string) "STRIP "STRING" TEST"

#define TEST04(A,B,C) A ##B #C
gCling->getMacroValue("TEST04")
// CHECK-NEXT: (std::string) "A ## B # C"

// expected-no-diagnostics
.q
