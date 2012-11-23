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

    FunctionDecl* FD = getTransaction()->getWrapperFD();

    int foundAtPos = -1;
    Expr* lastExpr = utils::Analyze::GetOrCreateLastExpr(FD, &foundAtPos, 
                                                         /*omitDS*/false,
                                                         m_Sema);
    if (lastExpr) {
      QualType RetTy = lastExpr->getType();
      if (!RetTy->isVoidType() && RetTy.isTriviallyCopyableType(*m_Context)) {
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
        // Change it to a return stmt (Avoid dealloc/alloc of all el.)
        *(CS->body_begin() + foundAtPos)
          = m_Sema->ActOnReturnStmt(lastExpr->getExprLoc(), 
                                    lastExpr).take();
      }
    } else if (foundAtPos >= 0) {
      // check for non-void return statement
      CompoundStmt* CS = cast<CompoundStmt>(FD->getBody());
      Stmt* CSS = *(CS->body_begin() + foundAtPos);
      if (ReturnStmt* RS = dyn_cast<ReturnStmt>(CSS)) {
        if (Expr* RetV = RS->getRetValue()) {
          QualType RetTy = RetV->getType();
          // Any return statement will have been "healed" by Sema
          // to correspond to the original void return type of the
          // wrapper, using a ImplicitCastExpr 'void' <ToVoid>.
          // Remove that.
          if (RetTy->isVoidType()) {
            ImplicitCastExpr* VoidCast = dyn_cast<ImplicitCastExpr>(RetV);
            if (VoidCast) {
              RS->setRetValue(VoidCast->getSubExpr());
              RetTy = VoidCast->getSubExpr()->getType();
            }
          }

          if (!RetTy->isVoidType()
              && RetTy.isTriviallyCopyableType(*m_Context)) {
            Sema::ContextRAII pushedDC(*m_Sema, FD);
            FunctionProtoType::ExtProtoInfo EPI;
            QualType FnTy = m_Context->getFunctionType(RetTy,
                                                       /* ArgArray = */0,
                                                       /* NumArgs = */0,
                                                       EPI);
            FD->setType(FnTy);
          } // not returning void
        } // have return value
      } // is a return statement
    } // have a statement
  }
} // end namespace cling
