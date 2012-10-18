// RUN: cat %s | %cling 2>&1 | FileCheck %s

#include "cling/Interpreter/Interpreter.h"

gCling->execute("1;");
extern "C" int printf(const char* fmt, ...);
gCling->execute("printf(\"%d\", printf(\"%d\",1));");
// CHECK: 11
