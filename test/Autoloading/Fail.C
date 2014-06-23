//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S -Xclang -verify
// Test templateFail
//XFAIL: *
//All the currently failing stuff

#include "cling/Interpreter/Interpreter.h"
// #include "cling/Interpreter/AutoloadCallback.h"
gCling->GenerateAutoloadingMap("Fail.h","test.h");

gCling->process("const char * const argV = \"cling\";");
gCling->process("cling::Interpreter *DefaultInterp;");

gCling->process("DefaultInterp = new cling::Interpreter(1, &argV);");
gCling->process("DefaultInterp->process(\"#include \\\"test.h\\\"\");");

.q
