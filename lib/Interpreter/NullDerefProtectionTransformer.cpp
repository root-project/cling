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

  class IfStmtInjector : public StmtVisitor<IfStmtInjector, void> {
  private:
    Sema& m_Sema;
    typedef std::map<clang::FunctionDecl*, std::bitset<32> > decl_map_t;
    std::map<clang::FunctionDecl*, std::bitset<32> > m_NonNullArgIndexs;

    ///\brief Needed for the AST transformations, owned by Sema.
    ///
    ASTContext& m_Context;

    ///\brief cling_runtime_internal_throwIfInvalidPointer cache.
    ///
    LookupResult* m_LookupResult;

  public:
    IfStmtInjector(Sema& S) : m_Sema(S), m_Context(S.getASTContext()),
    m_LookupResult(0) {}
    void Inject(CompoundStmt* CS) {
      VisitStmt(CS);
    }

    void VisitStmt(Stmt* S) {
      for (auto child: S->children()) {
        if (child)
          Visit(child);
      }
    }

    void VisitUnaryOperator(UnaryOperator* UnOp) {
      Visit(UnOp->getSubExpr());
      if (UnOp->getOpcode() == UO_Deref) {
        Expr* newSubExpr = SynthesizeCheck(UnOp->getLocStart(),
                                 UnOp->getSubExpr());
        UnOp->setSubExpr(newSubExpr);
      }
    }

    void VisitMemberExpr(MemberExpr* ME) {
      Visit(ME->getBase());
      if (ME->isArrow()) {
        Expr* newBase = SynthesizeCheck(ME->getLocStart(), ME->getBase());
        ME->setBase(newBase);
      }
    }

   void VisitCallExpr(CallExpr* CE) {
      Visit(CE->getCallee());
      FunctionDecl* FDecl = CE->getDirectCallee();
      if (FDecl && isDeclCandidate(FDecl)) {
        decl_map_t::const_iterator it = m_NonNullArgIndexs.find(FDecl);
        const std::bitset<32>& ArgIndexs = it->second;
        Sema::ContextRAII pushedDC(m_Sema, FDecl);
        for (int index = 0; index < 32; ++index) {
          if (ArgIndexs.test(index)) {
            // Get the argument with the nonnull attribute.
            Expr* Arg = CE->getArg(index);
            Expr* newArg = SynthesizeCheck(Arg->getLocStart(), Arg);
            CE->setArg(index, newArg);
          }
        }
      }
    }

  private:
    Expr* SynthesizeCheck(SourceLocation Loc, Expr* Arg) {
      assert(Arg && "Cannot call with Arg=0");

      if(!m_LookupResult)
        FindAndCacheRuntimeLookupResult();

      Expr* VoidSemaArg = utils::Synthesize::CStyleCastPtrExpr(&m_Sema,
                                                            m_Context.VoidPtrTy,
                                                            (uint64_t)&m_Sema);

      Expr* VoidExprArg = utils::Synthesize::CStyleCastPtrExpr(&m_Sema,
                                                          m_Context.VoidPtrTy,
                                                          (uint64_t)Arg);

      Expr *args[] = {VoidSemaArg, VoidExprArg, Arg};

      Scope* S = m_Sema.getScopeForContext(m_Sema.CurContext);

      CXXScopeSpec CSS;
      Expr* unresolvedLookup
        = m_Sema.BuildDeclarationNameExpr(CSS, *m_LookupResult,
                                         /*ADL*/ false).get();

      Expr* call = m_Sema.ActOnCallExpr(S, unresolvedLookup, Loc,
                                        args, Loc).get();

      TypeSourceInfo* TSI
              = m_Context.getTrivialTypeSourceInfo(Arg->getType(), Loc);
      Expr* castExpr = m_Sema.BuildCStyleCastExpr(Loc, TSI, Loc, call).get();

      return castExpr;
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

    void FindAndCacheRuntimeLookupResult() {
      assert(!m_LookupResult && "Called multiple times!?");

      DeclarationName Name
        = &m_Context.Idents.get("cling_runtime_internal_throwIfInvalidPointer");

      SourceLocation noLoc;
      m_LookupResult = new LookupResult(m_Sema, Name, noLoc,
                                        Sema::LookupOrdinaryName,
                                        Sema::ForRedeclaration);
      m_Sema.LookupQualifiedName(*m_LookupResult,
                                 m_Context.getTranslationUnitDecl());
      assert(!m_LookupResult->empty() &&
              "cling_runtime_internal_throwIfInvalidPointer");
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
    injector.Inject(CS);
    FD->setBody(CS);
    return Result(FD, true);
  }
} // end namespace cling
