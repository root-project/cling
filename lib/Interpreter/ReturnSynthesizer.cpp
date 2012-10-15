//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ReturnSynthesizer.h"

#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

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

          int foundAtPos = -1;
          if (Expr* lastExpr = utils::Analyze::GetLastExpr(FD, &foundAtPos)) {
            QualType RetTy = lastExpr->getType();
            if (!RetTy->isVoidType()) {
              // Change the void function's return type
              // We can't PushDeclContext, because we don't have scope.
              Sema::ContextRAII pushedDC(*m_Sema, FD);
              FunctionProtoType::ExtProtoInfo EPI;
              QualType FnTy = m_Context->getFunctionType(RetTy,
                                                         /* ArgArray = */0,
                                                         /* NumArgs = */0, EPI);
              FD->setType(FnTy);
              CompoundStmt* CS = cast<CompoundStmt>(FD->getBody());
              assert(CS && "Missing body?");
              // Change it with return stmt (Avoid dealloc/alloc of all el.)
              *(CS->body_begin() + foundAtPos)
                = m_Sema->ActOnReturnStmt(lastExpr->getExprLoc(), 
                                            lastExpr).take();
            }
          }
        }
  }
} // end namespace cling
