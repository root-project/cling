//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Baozeng Ding <sploving1@gmail.com>
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ASTNullDerefProtection.h"

#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"

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

  Expr* ASTNullDerefProtection::InsertThrow(SourceLocation* Loc, Expr* Arg) {
    ASTContext* Context = &m_Sema->getASTContext();
    Scope* S = m_Sema->getScopeForContext(m_Sema->CurContext);
    //copied from DynamicLookup.cpp
    NamespaceDecl* NSD = utils::Lookup::Namespace(m_Sema, "cling");
    NamespaceDecl* clingRuntimeNSD
      = utils::Lookup::Namespace(m_Sema, "runtime", NSD);

    // Find and set up "NullDerefException"
    DeclarationName Name
      = &Context->Idents.get("NullDerefException");

    LookupResult R(*m_Sema, Name, SourceLocation(),
      Sema::LookupOrdinaryName, Sema::ForRedeclaration);

    m_Sema->LookupQualifiedName(R, clingRuntimeNSD);
    CXXRecordDecl* NullDerefDecl = R.getAsSingle<CXXRecordDecl>();

    // Lookup Sema type
    CXXRecordDecl* SemaRD
      = dyn_cast<CXXRecordDecl>(utils::Lookup::Named(m_Sema, "Sema",
      utils::Lookup::Namespace(m_Sema, "clang")));

    QualType SemaRDTy = Context->getTypeDeclType(SemaRD);
    Expr* VoidSemaArg = utils::Synthesize::CStyleCastPtrExpr(m_Sema, SemaRDTy,
      (uint64_t)m_Sema);

    // Lookup Expr type
    CXXRecordDecl* ExprRD
      = dyn_cast<CXXRecordDecl>(utils::Lookup::Named(m_Sema, "Expr",
      utils::Lookup::Namespace(m_Sema, "clang")));

    QualType ExprRDTy = Context->getTypeDeclType(ExprRD);
    Expr* VoidExprArg = utils::Synthesize::CStyleCastPtrExpr(m_Sema, ExprRDTy,
      (uint64_t)Arg);

    Expr *args[] = {VoidSemaArg, VoidExprArg};
    QualType NullDerefDeclTy = Context->getTypeDeclType(NullDerefDecl);
    TypeSourceInfo* TSI = Context->getTrivialTypeSourceInfo(NullDerefDeclTy, 
                                                            SourceLocation());
    ExprResult ConstructorCall
      = m_Sema->BuildCXXTypeConstructExpr(TSI,*Loc, MultiExprArg(args, 2),*Loc);

    ExprResult Throw = m_Sema->ActOnCXXThrow(S, *Loc, ConstructorCall.take());
    return Throw.get();
  }


  Stmt* ASTNullDerefProtection::SynthesizeCheck(SourceLocation Loc, Expr* Arg ){
    Expr* Throw = InsertThrow(&Loc, Arg);
    ASTContext& Context = m_Sema->getASTContext();
    // Check whether we can get the argument'value. If the argument is
    // null, throw an exception direclty. If the argument is not null
    // then ignore this argument and continue to deal with the next
    // argument with the nonnull attribute.
    bool Result = false;
    if (Arg->EvaluateAsBooleanCondition(Result, Context)) {
      if(!Result) {
        return Throw;
      }
      return Arg;
    }
    // The argument's value cannot be decided, so we add a UnaryOp
    // operation to check its value at runtime.
    DeclRefExpr* DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts());
    assert(DRE && "No declref expr?");
    Scope* S = m_Sema->getScopeForContext(m_Sema->CurContext);
    ExprResult ER = m_Sema->ActOnUnaryOp(S, Loc, tok::exclaim, DRE);

    Decl* varDecl = 0;
    Stmt* varStmt = 0;
    Sema::FullExprArg FullCond(m_Sema->MakeFullExpr(ER.take()));
    StmtResult IfStmt = m_Sema->ActOnIfStmt(Loc, FullCond, varDecl,
                                            Throw, Loc, varStmt);
    return IfStmt.take();
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
    SourceLocation Loc = FD->getBody()->getLocStart();

    for (CompoundStmt::body_iterator I = CS->body_begin(), EI = CS->body_end();
         I != EI; ++I) {
      CallExpr* CE = dyn_cast<CallExpr>(*I);
      if (CE) {
        if (FunctionDecl* FDecl = CE->getDirectCallee()) {
          if(FDecl && isDeclCandidate(FDecl)) {
            SourceLocation CallLoc = CE->getLocStart();
            decl_map_t::const_iterator it = m_NonNullArgIndexs.find(FDecl);
            const std::bitset<32>& ArgIndexs = it->second;
            Sema::ContextRAII pushedDC(*m_Sema, FDecl);
            for (int index = 0; index < 32; ++index) {
              if (!ArgIndexs.test(index)) continue;

              // Get the argument with the nonnull attribute.
              Expr* Arg = CE->getArg(index);
              
              Stmts.push_back(SynthesizeCheck(CallLoc, Arg));
            }
          }
        }
        //Stmts.push_back(CE);
      }
      else {
        if (BinaryOperator* BinOp = dyn_cast<BinaryOperator>(*I)) {
          SourceLocation SL = BinOp->getLocStart();
          Expr* LHS = BinOp->getLHS()->IgnoreImpCasts();
          Expr* RHS = BinOp->getRHS()->IgnoreImpCasts();
          UnaryOperator* LUO = dyn_cast<UnaryOperator>(LHS);
          UnaryOperator* RUO = dyn_cast<UnaryOperator>(RHS);
          if (LUO  && LUO->getOpcode() == UO_Deref) {
            Expr* Op = dyn_cast<DeclRefExpr>(LUO->getSubExpr()->IgnoreImpCasts());
            if (Op) {
              bool Result = false;
              if (Op->EvaluateAsBooleanCondition(Result, *Context)) {
                if(!Result) {
                  Expr* Throw = InsertThrow(&SL, Op);
                  Stmts.push_back(Throw);
                }
              }
              else {
                Expr* Throw = InsertThrow(&SL, Op);
                ExprResult ER
                 = m_Sema->Sema::ActOnUnaryOp(S, SL, tok::exclaim, Op);

                Decl* varDecl = 0;
                Stmt* varStmt = 0;

                Sema::FullExprArg FullCond(m_Sema->MakeFullExpr(ER.take()));
                StmtResult IfStmt = m_Sema->ActOnIfStmt(SL, FullCond, varDecl,
                                                 Throw, SL, varStmt);
                Stmts.push_back(IfStmt.get());
              }
            }
          }
          if (RUO  && RUO->getOpcode() == UO_Deref) {
            Expr* Op = dyn_cast<DeclRefExpr>(RUO->getSubExpr()->IgnoreImpCasts());
            if (Op) {
              bool Result = false;
              if (Op->EvaluateAsBooleanCondition(Result, *Context)) {
                if(!Result) {
                  Expr* Throw = InsertThrow(&SL, Op);
                  Stmts.push_back(Throw);
                }
              }
              else {
                Stmts.push_back(SynthesizeCheck(SL, Op));
              }
            }
          }
        }
      }
      Stmts.push_back(*I);
    }
    llvm::ArrayRef<Stmt*> StmtsRef(Stmts.data(), Stmts.size());
    CompoundStmt* CSBody = new (*Context)CompoundStmt(*Context, StmtsRef,
                                                           Loc, Loc);
    FD->setBody(CSBody);
    DC->removeDecl(FD);
    DC->addDecl(FD);
  }
} // end namespace cling
