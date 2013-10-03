// RUN: cat %s | %cling 2>&1 | FileCheck %s
// Test findArgList 

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/SmallVector.h"

.rawInput 1
int a;
template<typename T> class aClass {
  T f() {
    return T();
  }
};
.rawInput 0
const cling::LookupHelper& lookup = gCling->getLookupHelper();
llvm::SmallVector<clang::Expr*, 4> exprs;
std::string buf;
clang::PrintingPolicy Policy(gCling->getSema().getASTContext().getPrintingPolicy());

lookup.findArgList("a, a", exprs);

exprs[0]->dumpPretty(gCling->getSema().getASTContext());
//CHECK: a
exprs[1]->dumpPretty(gCling->getSema().getASTContext());
//CHECK: a
.q
