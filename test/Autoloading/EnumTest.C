//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S -Xclang -verify
// Test EnumTest
//XFAIL: *

#include "cling/Interpreter/Interpreter.h"
gCling->GenerateAutoloadingMap("Enum.h","test.h");
.undo 1
#include "test.h"
#include "Enum.h"

//expected-no-diagnostics

.q