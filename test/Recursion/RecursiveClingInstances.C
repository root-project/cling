// RUN: cat %s | %cling -I%p 2>&1

// XFAIL:*
// XFAIL because there is no such function gCling->createUniqueName() (taking)
// 0 args. If run in gdb it reports the error. If run with the testsuite it say
// success. Probably system i/o is not properly piped.

// Tests the ability of cling to host itself. We can have cling instances in
// cling's runtime. This is important for people, who use cling embedded in
// their frameworks.

#include "cling/Interpreter/Interpreter.h"

gCling->process("const char * const argV = \"cling\";");
gCling->process("cling::Interpreter *DefaultInterp;");

gCling->process("DefaultInterp = new cling::Interpreter(1, &argV);");
gCling->process("DefaultInterp->process(\"#include \\\"cling/Interpreter/Interpreter.h\\\"\");");
gCling->process("DefaultInterp->process(\"gCling->createUniqueName()\");");
