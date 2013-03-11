//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ASTDumper.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {

  // pin the vtable to this file
  ASTDumper::~ASTDumper() {}


  void ASTDumper::Transform() {
    if (!getTransaction()->getCompilationOpts().Debug)
      return;

    // FIXME: Size might change in the loop!
    for (size_t Idx = 0; Idx < getTransaction()->size(); ++Idx) {
       // Copy DCI; it might get relocated below.
      Transaction::DelayCallInfo I = (*getTransaction())[Idx];
      for (DeclGroupRef::const_iterator J = I.m_DGR.begin(), 
             JE = I.m_DGR.end(); J != JE; ++J)
        printDecl(*J);
    }
  }

  void ASTDumper::printDecl(Decl* D) {
    PrintingPolicy Policy = D->getASTContext().getPrintingPolicy();
    if (m_Dump)
      Policy.DumpSourceManager = &m_Sema->getSourceManager();
    else
      Policy.DumpSourceManager = 0;

    if (D) {
      llvm::errs() << "\n-------------------Declaration---------------------\n";
      D->dump();

      if (Stmt* Body = D->getBody()) {
        llvm::errs() << "\n------------------Declaration Body---------------\n";
        Body->dump();
      }
      llvm::errs() << "\n---------------------------------------------------\n";
    }
  }
} // namespace cling
