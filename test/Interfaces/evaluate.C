//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify | FileCheck %s

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"
#include <cmath>

cling::Value V;
V // CHECK: (cling::Value &) <<<invalid>>> @0x{{.*}}

gCling->evaluate("return 1;", V);
V // CHECK: (cling::Value &) boxes [(int) 1]

gCling->evaluate("(void)V", V);
V // CHECK-NEXT: (cling::Value &) boxes [(void) ]

// Returns must put the result in the Value.
bool cond = true;
gCling->evaluate("if (cond) return \"true\"; else return 0;", V);
V // CHECK-NEXT: (cling::Value &) boxes [(const char [5]) "true"]
gCling->evaluate("if (cond) return; else return 12;", V);
V // CHECK-NEXT: (cling::Value &) boxes [(void) ]
gCling->evaluate("if (cond) return; int aa = 12;", V);
V // CHECK-NEXT: (cling::Value &) boxes [(void) ]
gCling->evaluate("cond = false; if (cond) return \"true\"; else return 0;", V);
V // CHECK-NEXT: (cling::Value &) boxes [(int) 0]

gCling->evaluate("bool a = [](){return true;};", V);
V // CHECK-NEXT: (cling::Value &) boxes [(bool) 1]

gCling->evaluate("auto a = 12.3; a;", V);
V // CHECK: (cling::Value &) boxes [(double) 1.230000e+01]

long LongV = 17;
gCling->evaluate("LongV;", V);
V // CHECK: (cling::Value &) boxes [(long) 17]

int* IntP = (int*)0x12;
gCling->evaluate("IntP;", V);
V // CHECK: (cling::Value &) boxes [(int *) 0x12]

cling::Value Result;
gCling->evaluate("V", Result);
// Here we check what happens for record type like cling::Value; they are returned by reference.
Result // CHECK: (cling::Value &) boxes [(cling::Value &) boxes [(int *) 0x12]]
V // CHECK: (cling::Value &) boxes [(int *) 0x12]

// Savannah #96277
gCling->evaluate("gCling->declare(\"double sin(double);\"); double one = sin(3.141/2);", V);
V // CHECK: (cling::Value &) boxes [(double) 1.000000e+00]

gCling->process("double one = sin(3.141/2);", &V);
V // CHECK: (cling::Value &) boxes [(double) 1.000000e+00]
one // CHECK: (double) 1.000
int one; // expected-error {{redefinition of 'one' with a different type: 'int' vs 'double'}} expected-note {{previous definition is here}}

// Make sure that PR#98434 doesn't get reintroduced.
.rawInput
void f(int) { return; }
.rawInput

gCling->evaluate("f", V);
V.isValid() //CHECK: {{\([_]B|b}}ool) true
// end PR#98434

// Check lifetime of objects in Value
.rawInput 1
struct WithDtor {
   static int fgCount;
   WithDtor() { ++fgCount; }
   WithDtor(const WithDtor&) { ++fgCount; }
   ~WithDtor() { --fgCount; }
};
int WithDtor::fgCount = 0;
WithDtor getWithDtor() { return WithDtor(); }
#include <vector>
std::vector<WithDtor> getWithDtorVec() { std::vector<WithDtor> ret; ret.resize(7); return ret; }
.rawInput 0

cling::Value* VOnHeap = new cling::Value();
gCling->evaluate("getWithDtor()", *VOnHeap);
*VOnHeap //CHECK: (cling::Value &) boxes [(WithDtor) @0x{{.*}}]
WithDtor::fgCount //CHECK: (int) 1
delete VOnHeap;
WithDtor::fgCount //CHECK: (int) 0

// Check destructor call for templates
VOnHeap = new cling::Value();
gCling->evaluate("getWithDtorVec()", *VOnHeap);
*VOnHeap //CHECK: (cling::Value &) boxes [(std::vector<WithDtor>) @0x{{.*}}]
WithDtor::fgCount //CHECK: (int) 7
delete VOnHeap;
WithDtor::fgCount //CHECK: (int) 0

// long doubles (tricky for the JIT).
gCling->evaluate("17.42L", V);
V // CHECK: (cling::Value &) boxes [(long double) 17.42{{[0-9]*}}L]

// Test references, temporaries
.rawInput 1
extern "C" int printf(const char*,...);
struct Tracer {
  std::string Content;
  static int InstanceCount;
  Tracer(const char* str): Content(str) { ++InstanceCount; dump("ctor"); }
  Tracer(const Tracer& o): Content(o.Content + "+") {
    ++InstanceCount; dump("copy");
  }
  ~Tracer() {--InstanceCount; dump("dtor");}
  std::string asStr() const {
    return Content + "{" + (char)('0' + InstanceCount) + "}";
  }
  void dump(const char* tag) { printf("%s:%s\n", asStr().c_str(), tag); }
};
int Tracer::InstanceCount = 0;

Tracer ObjMaker() { return Tracer("MADE"); }
Tracer& RefMaker() { static Tracer R("REF"); return R; }
const Tracer& ConstRefMaker() {static Tracer R("CONSTREF"); return R;}
namespace cling {
  // FIXME: inline printValue is not used by PrintClingValue()!
  std::string printValue(const Tracer* const p, const Tracer* const u,
                         const Value& V) {
    return p->asStr();
  }
}
void dumpTracerSVR(cling::Value& svr) {
  ((Tracer*)svr.getAs<void*>())->dump("dump");
}
.rawInput 0

// Creating the static in constructs one object. It gets returned by
// reference; it should only be destructed by ~JIT, definitely not by
// ~Value (which should only store a Tracer&)
gCling->evaluate("RefMaker()", V);
// This is the local static:
// CHECK: REF{1}:ctor
printf("RefMaker() done\n"); // CHECK-NEXT: RefMaker() done
V // CHECK-NEXT: (cling::Value &) boxes [(Tracer &) @{{.*}}]
dumpTracerSVR(V); // CHECK-NEXT: REF{1}:dump

// Setting a new value should destruct the old - BUT it's a ref thus no
// destruction.

// Create a temporary. Copies it into V through placement-new and copy
// construction. The latter is elided; the temporary *is* what's stored in V.
// Thus all we see is the construction of the temporary.
gCling->evaluate("ObjMaker()", V);
// The temporary gets created:
// CHECK-NEXT:MADE{2}:ctor
printf("ObjMaker() done\n"); //CHECK-NEXT: ObjMaker() done
V // CHECK-NEXT: (cling::Value &) boxes [(Tracer) @{{.*}}]
dumpTracerSVR(V); // CHECK-NEXT: MADE{2}:dump

// Creating a variable:
Tracer RT("VAR"); // CHECK-NEXT: VAR{3}:ctor

// The following is a declRefExpr of lvalue type. We explicitly treat this as
// a reference; i.e. the cling::Value will claim to store a Tracer&. No extra
// construction, no extra allocation should happen.
//
// Setting a new value should destruct the old:
// CHECK-NEXT: MADE{2}:dtor
gCling->evaluate("RT", V); // should not call any ctor!
printf("RT done\n"); //CHECK-NEXT: RT done
V // CHECK-NEXT: (cling::Value &) boxes [(Tracer &) @{{.*}}]
dumpTracerSVR(V); // CHECK-NEXT: VAR{2}:dump

// The following creates a copy, explicitly. This temporary object is then put
// into the Value.
//
gCling->evaluate("(Tracer)RT", V);
// Copies RT:
//CHECK-NEXT: VAR+{3}:copy
printf("(Tracer)RT done\n"); //CHECK-NEXT: RT done
V // CHECK-NEXT: (cling::Value &) boxes [(Tracer) @{{.*}}]
dumpTracerSVR(V); // CHECK-NEXT: VAR+{3}:dump

// Check eval of array var
Tracer arrV[] = {ObjMaker(), ObjMaker(), ObjMaker()};
// The array is built:
//CHECK-NEXT: MADE{4}:ctor
//CHECK-NEXT: MADE{5}:ctor
//CHECK-NEXT: MADE{6}:ctor

gCling->evaluate("arrV", V);
// Now V gets destructed...
//CHECK-NEXT: VAR+{5}:dtor
// ...and the elements are copied:
//CHECK-NEXT: MADE+{6}:copy
//CHECK-NEXT: MADE+{7}:copy
//CHECK-NEXT: MADE+{8}:copy

V // CHECK-NEXT: (cling::Value &) boxes [(Tracer [3]) { @{{.*}}, @{{.*}}, @{{.*}} }]

// Destruct the variables with static storage:
// Destruct arrV:
//CHECK-NEXT: MADE{7}:dtor
//CHECK-NEXT: MADE{6}:dtor
//CHECK-NEXT: MADE{5}:dtor

// CHECK-NEXT: VAR{4}:dtor
// CHECK-NEXT: REF{3}:dtor

//CHECK-NEXT: MADE+{2}:dtor
//CHECK-NEXT: MADE+{1}:dtor
//CHECK-NEXT: MADE+{0}:dtor
