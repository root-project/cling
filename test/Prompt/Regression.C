//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p | FileCheck %s

// This file should be used as regression test for the meta processing subsystem
// Reproducers of fixed bugs should be put here

// PR #96277
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"
#include <stdio.h>
gCling->declare("int print() { printf(\"print is run.\\n\"); return 1; }");
cling::Value V;
gCling->process("int a = print();",&V);
//CHECK: print is run.
gCling->process("a", &V);
//CHECK: (int) 1
gCling->process("a;", &V);
//CHECK-NOT: print is run.
// End PR #96277
// PR #98146
gCling->process("\"Root\"", &V);
// CHECK: (const char [5]) "Root"
V
// CHECK: (cling::Value &) boxes [(const char [5]) "Root"]
// End PR #98146

.rawInput 1
typedef enum {k1 = 0, k2} enumName;
enumName var = k1;
.rawInput 0
var
// CHECK: (enumName) (k1) : ({{(unsigned )?}}int) 0
const enumName constVar = (enumName) 1 // k2 is invisible!
// CHECK: (const enumName) (k2) : ({{(unsigned )?}}int) 1

// ROOT-8036: check that library symbols do not override interpreter symbols
int step = 10 // CHECK: (int) 10
step // CHECK: (int) 10

gCling->process("#ifdef __UNDEFINED__\n42\n#endif")
//CHECK: (cling::Interpreter::CompilationResult) (cling::Interpreter::CompilationResult::kSuccess) : ({{(unsigned )?}}int) 0

// User input variants of above:
#ifdef NOTDEFINED
 gCling->echo("9")
#else
 gCling->echo("12")
#endif
//CHECK: (int) 12
//CHECK: (cling::Interpreter::CompilationResult) (cling::Interpreter::CompilationResult::kSuccess) : ({{(unsigned )?}}int) 0

#ifdef __CLING__
 gCling->echo("19");
#else
 gCling->echo("156")
#endif
//CHECK: (int) 19

// ROOT-8300
struct B { static void *fgStaticVar; B(){ printf("B::B()\n"); } };
B b; // CHECK: B::B()

// ROOT-7857
template <class T> void tfunc(T) {}
struct ROOT7857{
  void func() { tfunc((ROOT7857*)0); }
};
ROOT7857* root7857;

// ROOT-5248
class MyClass;
extern MyClass* my;
class MyClass {public: MyClass* getMyClass() {return 0;}} cl;
MyClass* my = cl.getMyClass();

//
printf("Auto flush printf\n");
//CHECK-NEXT: Auto flush printf
cout << "Auto flush cout\n";
//CHECK-NEXT: Auto flush cout
printf("Must flush print\n"); cout.flush();
//CHECK-NEXT: Must flush printf

.q
