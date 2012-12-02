// RUN: cat %s | %cling 2>&1 | FileCheck %s
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


const clang::ClassTemplateDecl* tmplt_out = lookup.findClassTemplate("TmpltOutside");

printf("tmplt_out: 0x%lx\n", (unsigned long) tmplt_out);
//CHECK: tmplt_out: 0x{{[1-9a-f][0-9a-f]*$}}
tmplt_out->getQualifiedNameAsString().c_str()
//CHECK-NEXT: (const char *) "TmpltOutside"


const clang::ClassTemplateDecl* tmplt_inside = lookup.findClassTemplate("OuterClass::TmpltInside");

printf("tmplt_inside: 0x%lx\n", (unsigned long) tmplt_out);
//CHECK: tmplt_inside: 0x{{[1-9a-f][0-9a-f]*$}}
tmplt_inside->getQualifiedNameAsString().c_str()
//CHECK-NEXT: (const char *) "OuterClass::TmpltInside"


const clang::ClassTemplateDecl* tmplt_vec = lookup.findClassTemplate("std::vector");

printf("tmplt_vec: 0x%lx\n", (unsigned long) tmplt_vec);
//CHECK: tmplt_vec: 0x{{[1-9a-f][0-9a-f]*$}}
tmplt_vec->getQualifiedNameAsString().c_str()
//CHECK-NEXT: (const char *) "std::vector"

