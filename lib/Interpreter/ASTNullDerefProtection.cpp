//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Baozeng Ding <sploving1@gmail.com>
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ASTNullDerefProtection.h"

#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {
  ASTNullDerefProtection::ASTNullDerefProtection(clang::Sema* S)
    : TransactionTransformer(S) {
  }

  ASTNullDerefProtection::~ASTNullDerefProtection()
  { }

  bool ASTNullDerefProtection::isDeclCandidate(FunctionDecl * FDecl) {
    if(m_NonNullArgIndexs.count(FDecl)) return true;

    std::bitset<32> ArgIndexs;
    for (specific_attr_iterator<NonNullAttr>
           I = FDecl->specific_attr_begin<NonNullAttr>(),
           E = FDecl->specific_attr_end<NonNullAttr>(); I != E; ++I) {

       NonNullAttr *NonNull = *I;
       for (NonNullAttr::args_iterator i = NonNull->args_begin(),
              e = NonNull->args_end(); i != e; ++i) {
          ArgIndexs.set(*i);
       }
    }

    if (ArgIndexs.any()) {
      m_NonNullArgIndexs.insert(std::make_pair(FDecl, ArgIndexs));
      return true;
    }
    return false;
  }

  void ASTNullDerefProtection::Transform() {

    FunctionDecl* FD = getTransaction()->getWrapperFD();
    if (!FD)
      return;

    CompoundStmt* CS = dyn_cast<CompoundStmt>(FD->getBody());
    assert(CS && "Function body not a CompundStmt?");

    Scope* S = m_Sema->getScopeForContext(m_Sema->CurContext);
    ASTContext* Context = &m_Sema->getASTContext();
    DeclContext* DC = FD->getTranslationUnitDecl();
    llvm::SmallVector<Stmt*, 4> Stmts;
    SourceLocation SL = FD->getBody()->getLocStart();

    for (CompoundStmt::body_iterator I = CS->body_begin(), EI = CS->body_end();
         I != EI; ++I) {
      CallExpr* CE = dyn_cast<CallExpr>(*I);
      if (!CE) {
        Stmts.push_back(*I);
        continue;
      }
      if (FunctionDecl* FDecl = CE->getDirectCallee()) {
        if(FDecl && isDeclCandidate(FDecl)) {
          SourceLocation SL = CE->getLocStart();
          decl_map_t::const_iterator it = m_NonNullArgIndexs.find(FDecl);
          const std::bitset<32>& ArgIndexs = it->second;
          Sema::ContextRAII pushedDC(*m_Sema, FDecl);
          for (int index = 0; index < 32; ++index) {
            if (!ArgIndexs.test(index)) continue;
            DeclRefExpr* DRE
              = dyn_cast<DeclRefExpr>(CE->getArg(index)->IgnoreImpCasts());
            if (!DRE) continue;
            ExprResult ER
              = m_Sema->Sema::ActOnUnaryOp(S, SL, tok::exclaim, DRE);

            IntegerLiteral* One = IntegerLiteral::Create(*Context,
              llvm::APInt(32, 1), Context->IntTy, SL);

            ExprResult Throw = m_Sema->ActOnCXXThrow(S, SL, One);
            Decl* varDecl = 0;
            Stmt* varStmt = 0;

            Sema::FullExprArg FullCond(m_Sema->MakeFullExpr(ER.take()));

            StmtResult IfStmt = m_Sema->ActOnIfStmt(SL, FullCond, varDecl,
                                                      Throw.get(), SL, varStmt);
            Stmts.push_back(IfStmt.get());
          }
        }
      }
      Stmts.push_back(CE);
    }
    llvm::ArrayRef<Stmt*> StmtsRef(Stmts.data(), Stmts.size());
    CompoundStmt* CSBody = new (*Context)CompoundStmt(*Context, StmtsRef,
                                                           SL, SL);
    FD->setBody(CSBody);
    DC->removeDecl(FD);
    DC->addDecl(FD);
  }
} // end namespace cling
