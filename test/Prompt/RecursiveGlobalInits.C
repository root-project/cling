//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s

// Checks for infinite recursion when we combine nested calls of process line
// with global initializers.

#include "cling/Interpreter/Interpreter.h"

class MyClass { public:  MyClass(){ gCling->process("gCling->getVersion()");} };

MyClass *My = new MyClass(); // CHECK: (const char *) "{{.*}}"

extern "C" int printf(const char*...);

// from https://sft.its.cern.ch/jira/browse/ROOT-5856?focusedCommentId=30869&page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel#comment-30869
struct S {
  S() {
    printf("Exec\n");
    gCling->process("printf(\"RecursiveExec\\n\");");
  }
} s;
// CHECK-NEXT:Exec
// CHECK-NEXT:RecursiveExec
gCling->process("int RecursiveInit = printf(\"A\\n\");");
int ForceInitSequence = 17;

// CHECK-NEXT:A

// CHECK-EMPTY:

.q
