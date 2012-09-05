// RUN: cat %s | %cling 2>&1 | FileCheck %s
// Test Interpreter::lookupType()
//
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "clang/AST/Type.h"

using namespace std;

//
//  Test Data.
//

.rawInput 1
class A {};
namespace N {
class B {};
namespace M {
class C {};
} // namespace M
} // namespace N
typedef int my_int;
.rawInput 0


const cling::LookupHelper& lookup = gCling->getLookupHelper();

clang::QualType cl_A = lookup.findType("A");
cl_A.getAsString().c_str()
//CHECK: (const char * const) "class A"

clang::QualType cl_B_in_N = lookup.findType("N::B");
cl_B_in_N.getAsString().c_str()
//CHECK: (const char * const) "N::B"

clang::QualType cl_C_in_M = lookup.findType("N::M::C");
cl_C_in_M.getAsString().c_str()
//CHECK: (const char * const) "N::M::C"

clang::QualType builtin_int = lookup.findType("int");
builtin_int.getAsString().c_str()
//CHECK: (const char * const) "int"

clang::QualType typedef_my_int = lookup.findType("my_int");
typedef_my_int.getAsString().c_str()
//CHECK: (const char * const) "my_int"

.q
