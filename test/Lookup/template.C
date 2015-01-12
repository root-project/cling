//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -fno-rtti 2>&1 | FileCheck %s
// Test findClassTemplate, which is esentially is a DeclContext.
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclTemplate.h"

#include <cstdio>
#include <vector>

using namespace std;
using namespace llvm;

.rawInput 1
class OuterClass {
public:
   template <typename T> class TmpltInside {};
};
template <typename T> class TmpltOutside {};
.rawInput 0

const cling::LookupHelper& lookup = gCling->getLookupHelper();
cling::LookupHelper::DiagSetting diags = cling::LookupHelper::WithDiagnostics;


const clang::ClassTemplateDecl* tmplt_out = lookup.findClassTemplate("TmpltOutside", diags);

printf("tmplt_out: 0x%lx\n", (unsigned long) tmplt_out);
//CHECK: tmplt_out: 0x{{[1-9a-f][0-9a-f]*$}}
tmplt_out->getQualifiedNameAsString().c_str()
//CHECK-NEXT: ({{[^)]+}}) "TmpltOutside"


const clang::ClassTemplateDecl* tmplt_inside = lookup.findClassTemplate("OuterClass::TmpltInside", diags);

printf("tmplt_inside: 0x%lx\n", (unsigned long) tmplt_out);
//CHECK: tmplt_inside: 0x{{[1-9a-f][0-9a-f]*$}}
tmplt_inside->getQualifiedNameAsString().c_str()
//CHECK-NEXT: ({{[^)]+}}) "OuterClass::TmpltInside"


const clang::ClassTemplateDecl* tmplt_vec = lookup.findClassTemplate("std::vector", diags);

printf("tmplt_vec: 0x%lx\n", (unsigned long) tmplt_vec);
//CHECK: tmplt_vec: 0x{{[1-9a-f][0-9a-f]*$}}
tmplt_vec->getQualifiedNameAsString().c_str()
//CHECK-NEXT: ({{[^)]+}}) "std::{{(__1::)?}}vector"

