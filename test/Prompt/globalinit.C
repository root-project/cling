// RUN: %cling %s\(\"%s\"\) | FileCheck %s

#include "cling/Interpreter/Interpreter.h"

void globalinit(const std::string& location) {
  gCling->loadFile(location + ".h", false); // CHECK: A::S()
  gCling->loadFile(location + "2.h", false); // CHECK: B::S()
}
// CHECK: B::~S()
// CHECK: A::~S()
