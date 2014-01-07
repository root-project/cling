//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
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

    Transaction* T = getTransaction();
    for (Transaction::const_iterator I = T->decls_begin(), E = T->decls_end();
         I != E; ++I) {
       // Copy DCI; it might get relocated below.
      Transaction::DelayCallInfo DCI = *I;
      for (DeclGroupRef::const_iterator J = DCI.m_DGR.begin(), 
             JE = DCI.m_DGR.end(); J != JE; ++J)
        printDecl(*J);
    }
  }

  void ASTDumper::printDecl(Decl* D) {
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
