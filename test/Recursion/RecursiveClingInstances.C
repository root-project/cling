// RUN: cat %s | %cling -I%p

// Tests the ability of cling to host itself. We can have cling instances in
// cling's runtime. This is important for people, who use cling embedded in
// their frameworks.

#include "cling/Interpreter/Interpreter.h"

gCling->process("const char * const argV = \"cling\";");
gCling->process("cling::Interpreter *DefaultInterp;");

gCling->process("DefaultInterp = new cling::Interpreter(1, &argV);");
gCling->process("DefaultInterp->process(\"#include \\\"cling/Interpreter/Interpreter.h\\\"\");");
gCling->process("DefaultInterp->process(\"gCling->createUniqueName()\");");
