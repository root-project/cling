//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -fno-rtti 2>&1 | FileCheck %s
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
using clang::QualType;
using cling::LookupHelper;
.rawInput 0


const LookupHelper& lookup = gCling->getLookupHelper();

QualType cl_A = lookup.findType("A", LookupHelper::WithDiagnostics);
cl_A.getAsString().c_str()
//CHECK: ({{[^)]+}}) "class A"

QualType cl_B_in_N = lookup.findType("N::B", LookupHelper::WithDiagnostics);
cl_B_in_N.getAsString().c_str()
//CHECK: ({{[^)]+}}) "class N::B"

QualType cl_C_in_M = lookup.findType("N::M::C", LookupHelper::WithDiagnostics);
cl_C_in_M.getAsString().c_str()
//CHECK: ({{[^)]+}}) "class N::M::C"

QualType builtin_int = lookup.findType("int", LookupHelper::WithDiagnostics);
builtin_int.getAsString().c_str()
//CHECK: ({{[^)]+}}) "int"

QualType typedef_my_int = lookup.findType("my_int", LookupHelper::WithDiagnostics);
typedef_my_int.getAsString().c_str()
//CHECK: ({{[^)]+}}) "my_int"

.q
