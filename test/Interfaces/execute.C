//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s

#include "cling/Interpreter/Interpreter.h"

gCling->execute("1;");
extern "C" int printf(const char* fmt, ...);
gCling->execute("printf(\"%d\", printf(\"%d\",1));");
// CHECK: 11
