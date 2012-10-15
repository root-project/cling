//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ReturnSynthesizer.h"

#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {
  ReturnSynthesizer::ReturnSynthesizer(clang::Sema* S)
    : TransactionTransformer(S), m_Context(&S->getASTContext()) {
  }

  // pin the vtable here.
  ReturnSynthesizer::~ReturnSynthesizer() 
  { }

  void ReturnSynthesizer::Transform() {
    if (!getTransaction()->getCompilationOpts().ResultEvaluation)
      return;

    for (Transaction::const_iterator iDGR = getTransaction()->decls_begin(), 
           eDGR = getTransaction()->decls_end(); iDGR != eDGR; ++iDGR)
      for (DeclGroupRef::const_iterator I = (*iDGR).begin(), E = (*iDGR).end();
           I != E; ++I)
        if (FunctionDecl* FD = dyn_cast<FunctionDecl>(*I)) {
          if (FD->getNameAsString().find("__cling_Un1Qu3"))
            return;
          if (CompoundStmt* CS = dyn_cast<CompoundStmt>(FD->getBody())) {
            // Collect all Stmts, contained in the CompoundStmt
            llvm::SmallVector<Stmt *, 4> Stmts;
            for (CompoundStmt::body_iterator iStmt = CS->body_begin(),
                   eStmt = CS->body_end(); iStmt != eStmt; ++iStmt)
              Stmts.push_back(*iStmt);

            int indexOfLastExpr = Stmts.size();
            while(indexOfLastExpr--) {
              // find the trailing expression statement (skip e.g. null statements)
              if (isa<Expr>(Stmts[indexOfLastExpr])) {
                // even if void: we found an expression
                break;
              }
            }
            
            // If no expressions found quit early.
            if (indexOfLastExpr < 0)
              return; 
            // We can't PushDeclContext, because we don't have scope.
            Sema::ContextRAII pushedDC(*m_Sema, FD);
            Expr* lastExpr = cast<Expr>(Stmts[indexOfLastExpr]); // It is an expr
            if (lastExpr) {
              QualType RetTy = lastExpr->getType();
              if (!RetTy->isVoidType()) {
                // Change the void function's return type
                FunctionProtoType::ExtProtoInfo EPI;
                QualType FnTy = m_Context->getFunctionType(RetTy,
                                                           /* ArgArray = */0,
                                                           /* NumArgs = */0, 
                                                           EPI);
                FD->setType(FnTy);
                
                // Change it with return stmt
                Stmts[indexOfLastExpr]
                  = m_Sema->ActOnReturnStmt(lastExpr->getExprLoc(), 
                                            lastExpr).take();
              }
              CS->setStmts(*m_Context, Stmts.data(), Stmts.size());
            }
          }
        }
  }
} // end namespace cling
