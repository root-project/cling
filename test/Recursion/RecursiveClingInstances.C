// RUN: cat %s | %cling -I%p 2>&1 | FileCheck %s

// Tests the ability of cling to host itself. We can have cling instances in
// cling's runtime. This is important for people, who use cling embedded in
// their frameworks.

#include "cling/Interpreter/Interpreter.h"

gCling->process("const char * const argV = \"cling\";");
gCling->process("cling::Interpreter *DefaultInterp;");

gCling->process("DefaultInterp = new cling::Interpreter(1, &argV);");
gCling->process("DefaultInterp->process(\"#include \\\"cling/Interpreter/Interpreter.h\\\"\");");
gCling->process("DefaultInterp->process(\"std::string s; gCling->createUniqueName(s); s.c_str()\");");
// CHECK: (const char *) "__cling_Un1Qu31"
.q
