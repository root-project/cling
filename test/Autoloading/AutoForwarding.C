//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S -Xclang -verify
// Test FwdPrinterTest


#include "cling/Interpreter/Interpreter.h"
// #include "cling/Interpreter/AutoloadCallback.h"
gCling->GenerateAutoloadingMap("Def2.h","test.h");

gCling->process("const char * const argV = \"cling\";");
gCling->process("cling::Interpreter *DefaultInterp;");

gCling->process("DefaultInterp = new cling::Interpreter(1, &argV);");
// gCling->process("DefaultInterp->setCallbacks(new cling::AutoloadCallback(DefaultInterp));")
gCling->process("DefaultInterp->process(\"#include \\\"test.h\\\"\");");
gCling->process("DefaultInterp->process(\"#include \\\"Def2.h\\\"\");");

//expected-no-diagnostics

.q