//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling -Xclang -verify %s | FileCheck %s
extern "C" int printf(const char*,...);

class Foo {
private:
  static int privateFunc() { return 42; }
};

#include "cling/Interpreter/Interpreter.h"
void compileFunc() {
  typedef int (*myFunc_t)(int);

  const char* myFuncCode = "extern \"C\" int myFunc(int arg) { \n"
    "printf(\"arg is %d\\n\", arg); return arg * arg; \n"
    "}";

  myFunc_t myFuncP = (myFunc_t) gCling->compileFunction("myFunc", myFuncCode);
  printf("myFunc returned %d\n", (*myFuncP)(12));
  //CHECK: arg is 12
  //CHECK: myFunc returned 144

  // Test ifUniq == true:
  const char* myFuncCode2 = "extern \"C\" int myFunc(int arg) { \n"
    "printf(\"arg is %d\\n\", arg); return -1; \n"
    "}";
  if (gCling->compileFunction("myFunc", myFuncCode2) == myFuncP) {
    printf("As expected, myFunc() did not change.\n");
    //CHECK: As expected, myFunc() did not change.
  }

  // Test withAccessControl == false:
  const char* myPrivFuncCode = "extern \"C\" int myPrivFunc(int) { \n"
    "printf(\"privateFunc() returns %d\\n\", Foo::privateFunc()); return -1; \n"
    "}";
  myFunc_t myPrivFuncP
    = (myFunc_t) gCling->compileFunction("myPrivFunc", myPrivFuncCode,
                                         false /*ifUniq*/,
                                         false /*withAccessControl*/);
  printf("myPrivFunc returned %d\n", (*myPrivFuncP)(13));
  //CHECK: privateFunc() returns 42
  //CHECK: myPrivFunc returned -1


  const char* myBadFuncCode = "extern \"C\" int myBadFunc(int) { \n"
    "return NOFUZZY; //expected-error@2 {{use of undeclared identifier 'NOFUZZY'}} \n"
    "}";

  if (!gCling->compileFunction("myBadFunc", myBadFuncCode)) {
    printf("As expected, myBadFunc did not compile\n");
    //CHECK: myBadFunc did not compile
  }
}
