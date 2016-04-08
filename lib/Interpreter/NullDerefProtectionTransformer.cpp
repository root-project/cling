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

#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include "clang/AST/Expr.h"

#include <bitset>
#include <map>

using namespace clang;

namespace cling {
  NullDerefProtectionTransformer::NullDerefProtectionTransformer(clang::Sema* S)
    : WrapperTransformer(S) {
  }

  NullDerefProtectionTransformer::~NullDerefProtectionTransformer()
  { }

  class PointerCheckInjector : public RecursiveASTVisitor<PointerCheckInjector> {
  private:
    Sema& m_Sema;
    typedef llvm::DenseMap<clang::FunctionDecl*, std::bitset<32> > decl_map_t;
    llvm::DenseMap<clang::FunctionDecl*, std::bitset<32> > m_NonNullArgIndexs;

    ///\brief Needed for the AST transformations, owned by Sema.
    ///
    ASTContext& m_Context;

    ///\brief cling_runtime_internal_throwIfInvalidPointer cache.
    ///
    LookupResult* m_clingthrowIfInvalidPointerCache;

  public:
    PointerCheckInjector(Sema& S) : m_Sema(S), m_Context(S.getASTContext()),
    m_clingthrowIfInvalidPointerCache(0) {}

    bool VisitStmt(Stmt* S) {
      for (auto child: S->children()) {
        if (child)
            VisitStmt(child);
      }
      return true;
    }

    bool VisitUnaryOperator(UnaryOperator* UnOp) {
      VisitStmt(UnOp->getSubExpr());
      if (UnOp->getOpcode() == UO_Deref)
        if (UnOp->getSubExpr()->getType().getTypePtr()->isPointerType())
          UnOp->setSubExpr(SynthesizeCheck(UnOp->getSubExpr()));
      return true;
    }

    bool VisitMemberExpr(MemberExpr* ME) {
      VisitStmt(ME->getBase());
      if (ME->isArrow())
        ME->setBase(SynthesizeCheck(ME->getBase()));
      return true;
    }

    bool VisitCallExpr(CallExpr* CE) {
      VisitStmt(CE->getCallee());
      FunctionDecl* FDecl = CE->getDirectCallee();
      if (FDecl && isDeclCandidate(FDecl)) {
        decl_map_t::const_iterator it = m_NonNullArgIndexs.find(FDecl);
        const std::bitset<32>& ArgIndexs = it->second;
        Sema::ContextRAII pushedDC(m_Sema, FDecl);
        for (int index = 0; index < 32; ++index) {
          if (ArgIndexs.test(index)) {
            // Get the argument with the nonnull attribute.
            Expr* Arg = CE->getArg(index);
            if (Arg->getType().getTypePtr()->isPointerType())
              CE->setArg(index, SynthesizeCheck(Arg));
          }
        }
      }
      return true;
    }

    bool VisitFunctionDecl(FunctionDecl* FD) {
      if (!FD->isConstexpr()) {
        CompoundStmt* CS = dyn_cast_or_null<CompoundStmt>(FD->getBody());
        if (CS)
          VisitStmt(CS);
      }
      return true;
    }

    bool VisitCXXMethodDecl(CXXMethodDecl* FD) {
      if (!FD->isConstexpr()) {
        CompoundStmt* CS = dyn_cast_or_null<CompoundStmt>(FD->getBody());
        if (CS)
        VisitStmt(CS);
    }
      return true;
    }

    bool TraverseFunctionDecl(FunctionDecl* FD) {
      if (!FD->isConstexpr()) {
        CompoundStmt* CS = dyn_cast_or_null<CompoundStmt>(FD->getBody());
        if (CS)
          TraverseStmt(CS);
      }
      return true;
    }

    bool TraverseCXXMethodDecl(CXXMethodDecl* CMD) {
      if (!CMD->isConstexpr()){
        CompoundStmt* CS = dyn_cast_or_null<CompoundStmt>(CMD->getBody());
        if (CS)
        TraverseStmt(CS);
    }
      return true;
    }

  private:
    Expr* SynthesizeCheck(Expr* Arg) {
      assert(Arg && "Cannot call with Arg=0");

      if(!m_clingthrowIfInvalidPointerCache)
        FindAndCacheRuntimeLookupResult();

      SourceLocation Loc = Arg->getLocStart();
      Expr* VoidSemaArg = utils::Synthesize::CStyleCastPtrExpr(&m_Sema,
                                                            m_Context.VoidPtrTy,
                                                            (uint64_t)&m_Sema);
      Expr* VoidExprArg = utils::Synthesize::CStyleCastPtrExpr(&m_Sema,
                                                          m_Context.VoidPtrTy,
                                                          (uint64_t)Arg);
      Scope* S = m_Sema.getScopeForContext(m_Sema.CurContext);
      CXXScopeSpec CSS;

      Expr* checkCall
        = m_Sema.BuildDeclarationNameExpr(CSS,
                                          *m_clingthrowIfInvalidPointerCache,
                                         /*ADL*/ false).get();
      const clang::FunctionProtoType* checkCallType
        = llvm::dyn_cast<const clang::FunctionProtoType>(
            checkCall->getType().getTypePtr());

      TypeSourceInfo* constVoidPtrTSI = m_Context.getTrivialTypeSourceInfo(
        checkCallType->getParamType(2), Loc);

      Expr* voidPtrArg
        = m_Sema.BuildCStyleCastExpr(Loc, constVoidPtrTSI, Loc,
                                     Arg).get();

      Expr *args[] = {VoidSemaArg, VoidExprArg, voidPtrArg};

      if (Expr* call = m_Sema.ActOnCallExpr(S, checkCall,
                                         Loc, args, Loc).get()) {
        clang::TypeSourceInfo* argTSI = m_Context.getTrivialTypeSourceInfo(
                                        Arg->getType(), Loc);
        Expr* castExpr = m_Sema.BuildCStyleCastExpr(Loc, argTSI, Loc, call).get();
        return castExpr;
      }
      return voidPtrArg;
    }

    bool isDeclCandidate(FunctionDecl * FDecl) {
      if (m_NonNullArgIndexs.count(FDecl))
        return true;

      if (llvm::isa<CXXRecordDecl>(FDecl))
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
      assert(!m_clingthrowIfInvalidPointerCache && "Called multiple times!?");

      DeclarationName Name
        = &m_Context.Idents.get("cling_runtime_internal_throwIfInvalidPointer");
      SourceLocation noLoc;
      m_clingthrowIfInvalidPointerCache = new LookupResult(m_Sema, Name, noLoc,
                                        Sema::LookupOrdinaryName,
                                        Sema::ForRedeclaration);
      m_Sema.LookupQualifiedName(*m_clingthrowIfInvalidPointerCache,
                                 m_Context.getTranslationUnitDecl());
      assert(!m_clingthrowIfInvalidPointerCache->empty() &&
              "Lookup of cling_runtime_internal_throwIfInvalidPointer failed!");
    }
  };

  ASTTransformer::Result
  NullDerefProtectionTransformer::Transform(clang::Decl* D) {

    PointerCheckInjector injector(*m_Sema);
    injector.TraverseDecl(D);
    return Result(D, true);
  }
} // end namespace cling
