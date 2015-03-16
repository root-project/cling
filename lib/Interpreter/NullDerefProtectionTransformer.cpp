//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Baozeng Ding <sploving1@gmail.com>
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "NullDerefProtectionTransformer.h"

#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"

#include <bitset>
#include <map>

using namespace clang;

namespace cling {
  NullDerefProtectionTransformer::NullDerefProtectionTransformer(clang::Sema* S)
    : WrapperTransformer(S) {
  }

  NullDerefProtectionTransformer::~NullDerefProtectionTransformer()
  { }

  // Copied from clad - the clang/opencl autodiff project
  class NodeContext {
  public:
  private:
    typedef llvm::SmallVector<clang::Stmt*, 2> Statements;
    Statements m_Stmts;
  private:
    NodeContext() {};
  public:
    NodeContext(clang::Stmt* s) { m_Stmts.push_back(s); }
    NodeContext(clang::Stmt* s0, clang::Stmt* s1) {
      m_Stmts.push_back(s0);
      m_Stmts.push_back(s1);
    }

    bool isSingleStmt() const { return m_Stmts.size() == 1; }

    clang::Stmt* getStmt() {
      assert(isSingleStmt() && "Cannot get multiple stmts.");
      return m_Stmts.front();
    }
    const clang::Stmt* getStmt() const { return getStmt(); }
    const Statements& getStmts() const {
      return m_Stmts;
    }

    CompoundStmt* wrapInCompoundStmt(clang::ASTContext& C) const {
      assert(!isSingleStmt() && "Must be more than 1");
      llvm::ArrayRef<Stmt*> stmts
        = llvm::makeArrayRef(m_Stmts.data(), m_Stmts.size());
      clang::SourceLocation noLoc;
      return new (C) clang::CompoundStmt(C, stmts, noLoc, noLoc);
    }

    clang::Expr* getExpr() {
      assert(llvm::isa<clang::Expr>(getStmt()) && "Must be an expression.");
      return llvm::cast<clang::Expr>(getStmt());
    }
    const clang::Expr* getExpr() const {
      return getExpr();
    }

    void prepend(clang::Stmt* S) {
      m_Stmts.insert(m_Stmts.begin(), S);
    }

    void append(clang::Stmt* S) {
      m_Stmts.push_back(S);
    }
  };

  class IfStmtInjector : public StmtVisitor<IfStmtInjector, NodeContext> {
  private:
    Sema& m_Sema;
    typedef std::map<clang::FunctionDecl*, std::bitset<32> > decl_map_t;
    std::map<clang::FunctionDecl*, std::bitset<32> > m_NonNullArgIndexs;

  public:
    IfStmtInjector(Sema& S) : m_Sema(S) {}
    CompoundStmt* Inject(CompoundStmt* CS) {
      NodeContext result = VisitCompoundStmt(CS);
      return cast<CompoundStmt>(result.getStmt());
    }

    NodeContext VisitStmt(Stmt* S) {
      return NodeContext(S);
    }

    NodeContext VisitCompoundStmt(CompoundStmt* CS) {
      ASTContext& C = m_Sema.getASTContext();
      llvm::SmallVector<Stmt*, 16> stmts;
      for (CompoundStmt::body_iterator I = CS->body_begin(), E = CS->body_end();
           I != E; ++I) {
        NodeContext nc = Visit(*I);
        if (nc.isSingleStmt())
          stmts.push_back(nc.getStmt());
        else
          stmts.append(nc.getStmts().begin(), nc.getStmts().end());
      }

      llvm::ArrayRef<Stmt*> stmtsRef(stmts.data(), stmts.size());
      CompoundStmt* newCS = new (C) CompoundStmt(C, stmtsRef,
                                                 CS->getLBracLoc(),
                                                 CS->getRBracLoc());
      return NodeContext(newCS);
    }

    NodeContext VisitIfStmt(IfStmt* If) {
      NodeContext result(If);
      // check the condition
      NodeContext cond = Visit(If->getCond());
      if (!cond.isSingleStmt())
        result.prepend(cond.getStmts()[0]);
      return result;
    }

    NodeContext VisitCastExpr(CastExpr* CE) {
      NodeContext result = Visit(CE->getSubExpr());
      return result;
    }

    NodeContext VisitBinaryOperator(BinaryOperator* BinOp) {
      NodeContext result(BinOp);

      // Here we might get if(check) throw; binop rhs.
      NodeContext rhs = Visit(BinOp->getRHS());
      // Here we might get if(check) throw; binop lhs.
      NodeContext lhs = Visit(BinOp->getLHS());

      // Prepend those checks. It will become:
      // if(check_rhs) throw; if (check_lhs) throw; BinOp;
      if (!rhs.isSingleStmt()) {
        // FIXME:we need to loop from 0 to n-1
        result.prepend(rhs.getStmts()[0]);
      }
      if (!lhs.isSingleStmt()) {
        // FIXME:we need to loop from 0 to n-1
        result.prepend(lhs.getStmts()[0]);
      }
      return result;
    }

    NodeContext VisitUnaryOperator(UnaryOperator* UnOp) {
      NodeContext result(UnOp);
      if (UnOp->getOpcode() == UO_Deref) {
        result.prepend(SynthesizeCheck(UnOp->getLocStart(),
                                       UnOp->getSubExpr()));
      }
      return result;
    }

    NodeContext VisitMemberExpr(MemberExpr* ME) {
      NodeContext result(ME);
      if (ME->isArrow()) {
        result.prepend(SynthesizeCheck(ME->getLocStart(),
                                       ME->getBase()->IgnoreImplicit()));
      }
      return result;
    }

    NodeContext VisitCallExpr(CallExpr* CE) {
      FunctionDecl* FDecl = CE->getDirectCallee();
      NodeContext result(CE);
      if (FDecl && isDeclCandidate(FDecl)) {
        decl_map_t::const_iterator it = m_NonNullArgIndexs.find(FDecl);
        const std::bitset<32>& ArgIndexs = it->second;
        Sema::ContextRAII pushedDC(m_Sema, FDecl);
        for (int index = 0; index < 32; ++index) {
          if (ArgIndexs.test(index)) {
            // Get the argument with the nonnull attribute.
            Expr* Arg = CE->getArg(index);
            result.prepend(SynthesizeCheck(Arg->getLocStart(), Arg));
          }
        }
      }
      return result;
    }

  private:
    Stmt* SynthesizeCheck(SourceLocation Loc, Expr* Arg) {
      assert(Arg && "Cannot call with Arg=0");
      ASTContext& Context = m_Sema.getASTContext();
      //copied from DynamicLookup.cpp
      // Lookup Sema type
      CXXRecordDecl* SemaRD
        = dyn_cast<CXXRecordDecl>(utils::Lookup::Named(&m_Sema, "Sema",
                                   utils::Lookup::Namespace(&m_Sema, "clang")));

      QualType SemaRDTy = Context.getTypeDeclType(SemaRD);
      Expr* VoidSemaArg = utils::Synthesize::CStyleCastPtrExpr(&m_Sema,SemaRDTy,
                                                             (uint64_t)&m_Sema);

      // Lookup Expr type
      CXXRecordDecl* ExprRD
        = dyn_cast<CXXRecordDecl>(utils::Lookup::Named(&m_Sema, "Expr",
                                   utils::Lookup::Namespace(&m_Sema, "clang")));

      QualType ExprRDTy = Context.getTypeDeclType(ExprRD);
      Expr* VoidExprArg = utils::Synthesize::CStyleCastPtrExpr(&m_Sema,ExprRDTy,
                                                               (uint64_t)Arg);

      Expr *args[] = {VoidSemaArg, VoidExprArg};

      Scope* S = m_Sema.getScopeForContext(m_Sema.CurContext);
      DeclarationName Name
        = &Context.Idents.get("cling__runtime__internal__throwNullDerefException");

      SourceLocation noLoc;
      LookupResult R(m_Sema, Name, noLoc, Sema::LookupOrdinaryName,
                   Sema::ForRedeclaration);
      m_Sema.LookupQualifiedName(R, Context.getTranslationUnitDecl());
      assert(!R.empty() && "Cannot find valuePrinterInternal::Select(...)");

      CXXScopeSpec CSS;
      Expr* UnresolvedLookup
        = m_Sema.BuildDeclarationNameExpr(CSS, R, /*ADL*/ false).get();

      Expr* call = m_Sema.ActOnCallExpr(S, UnresolvedLookup, noLoc,
                                        args, noLoc).get();
      // Check whether we can get the argument'value. If the argument is
      // null, throw an exception direclty. If the argument is not null
      // then ignore this argument and continue to deal with the next
      // argument with the nonnull attribute.
      bool Result = false;
      if (Arg->EvaluateAsBooleanCondition(Result, Context)) {
        if(!Result) {
          return call;
        }
        return Arg;
      }
      // The argument's value cannot be decided, so we add a UnaryOp
      // operation to check its value at runtime.
      ExprResult ER = m_Sema.ActOnUnaryOp(S, Loc, tok::exclaim, Arg);

      Decl* varDecl = 0;
      Stmt* varStmt = 0;
      Sema::FullExprArg FullCond(m_Sema.MakeFullExpr(ER.get()));
      StmtResult IfStmt = m_Sema.ActOnIfStmt(Loc, FullCond, varDecl,
                                             call, Loc, varStmt);
      return IfStmt.get();
    }

    bool isDeclCandidate(FunctionDecl * FDecl) {
      if (m_NonNullArgIndexs.count(FDecl))
        return true;

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
  };

  ASTTransformer::Result
  NullDerefProtectionTransformer::Transform(clang::Decl* D) {
    FunctionDecl* FD = dyn_cast<FunctionDecl>(D);
    if (!FD || FD->isFromASTFile())
      return Result(D, true);

    CompoundStmt* CS = dyn_cast_or_null<CompoundStmt>(FD->getBody());
    if (!CS)
      return Result(D, true);

    IfStmtInjector injector(*m_Sema);
    CompoundStmt* newCS = injector.Inject(CS);
    FD->setBody(newCS);
    return Result(FD, true);
  }
} // end namespace cling
