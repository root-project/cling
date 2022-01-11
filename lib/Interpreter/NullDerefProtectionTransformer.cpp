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

#include "cling/Interpreter/Interpreter.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Lookup.h"

#include <bitset>

using namespace clang;

namespace {
using namespace cling;

class PointerCheckInjector : public RecursiveASTVisitor<PointerCheckInjector> {
  private:
    Interpreter& m_Interp;
    Sema& m_Sema;
    typedef llvm::DenseMap<clang::FunctionDecl*, std::bitset<32> > decl_map_t;
    llvm::DenseMap<clang::FunctionDecl*, std::bitset<32> > m_NonNullArgIndexs;

    ///\brief Needed for the AST transformations, owned by Sema.
    ///
    ASTContext& m_Context;

    ///\brief cling_runtime_internal_throwIfInvalidPointer cache.
    ///
    LookupResult* m_clingthrowIfInvalidPointerCache;

    bool IsTransparentThis(Expr* E) {
      if (llvm::isa<CXXThisExpr>(E))
        return true;
      if (auto ICE = dyn_cast<ImplicitCastExpr>(E))
        return IsTransparentThis(ICE->getSubExpr());
      return false;
    }

  public:
    PointerCheckInjector(Interpreter& I)
      : m_Interp(I), m_Sema(I.getCI()->getSema()),
        m_Context(I.getCI()->getASTContext()),
        m_clingthrowIfInvalidPointerCache(0) {}

    ~PointerCheckInjector() {
      delete m_clingthrowIfInvalidPointerCache;
    }

    bool VisitUnaryOperator(UnaryOperator* UnOp) {
      Expr* SubExpr = UnOp->getSubExpr();
      VisitStmt(SubExpr);
      if (UnOp->getOpcode() == UO_Deref
          && !IsTransparentThis(SubExpr)
          && SubExpr->getType().getTypePtr()->isPointerType())
          UnOp->setSubExpr(SynthesizeCheck(SubExpr));
      return true;
    }

    bool VisitMemberExpr(MemberExpr* ME) {
      Expr* Base = ME->getBase();
      VisitStmt(Base);
      if (ME->isArrow()
          && !IsTransparentThis(Base)
          && ME->getMemberDecl()->isCXXInstanceMember())
        ME->setBase(SynthesizeCheck(Base));
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
            if (Arg->getType().getTypePtr()->isPointerType()
                && !IsTransparentThis(Arg))
              CE->setArg(index, SynthesizeCheck(Arg));
          }
        }
      }
      return true;
    }

    bool TraverseFunctionDecl(FunctionDecl* FD) {
      // We cannot synthesize when there is a const expr
      // and if it is a function template (we will do the transformation on
      // the instance).
      if (!FD->isConstexpr() && !FD->getDescribedFunctionTemplate())
         RecursiveASTVisitor::TraverseFunctionDecl(FD);
      return true;
    }

    bool TraverseCXXMethodDecl(CXXMethodDecl* CXXMD) {
      // We cannot synthesize when there is a const expr.
      if (!CXXMD->isConstexpr())
        RecursiveASTVisitor::TraverseCXXMethodDecl(CXXMD);
      return true;
    }

  private:
    Expr* SynthesizeCheck(Expr* Arg) {
      assert(Arg && "Cannot call with Arg=0");

      if(!m_clingthrowIfInvalidPointerCache)
        FindAndCacheRuntimeLookupResult();

      SourceLocation Loc = Arg->getBeginLoc();
      Expr* VoidSemaArg = utils::Synthesize::CStyleCastPtrExpr(&m_Sema,
                                                            m_Context.VoidPtrTy,
                                                            (uintptr_t)&m_Interp);
      Expr* VoidExprArg = utils::Synthesize::CStyleCastPtrExpr(&m_Sema,
                                                          m_Context.VoidPtrTy,
                                                          (uintptr_t)Arg);
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

      // It is unclear whether this is the correct cast if the type
      // is dependent.  Hence, For now, we do not expect SynthesizeCheck to
      // be run on a function template.  It should be run only on function
      // instances.
      // When this is actually insert in a function template, it seems that
      // clang r272382 when instantiating the templates drops one of the part
      // of the implicit cast chain.
      // Namely in:
/*
`-ImplicitCastExpr 0x1010cea90 <col:4> 'const void *' <BitCast>
 `-ImplicitCastExpr 0x1026e0bc0 <col:4> 'const class TAttMarker *'
                    <UncheckedDerivedToBase (TAttMarker)>
  `-ImplicitCastExpr 0x1026e0b48 <col:4> 'class TGraph *' <LValueToRValue>
   `-DeclRefExpr 0x1026e0b20 <col:4> 'class TGraph *' lvalue Var 0x1026e09c0
                   'g5' 'class TGraph *'
*/
      // It drops the 2nd lines (ImplicitCastExpr UncheckedDerivedToBase)
      // clang r227800 seems to actually keep that lines during instantiation.
      Expr* voidPtrArg
        = m_Sema.BuildCStyleCastExpr(Loc, constVoidPtrTSI, Loc, Arg).get();

      Expr *args[] = {VoidSemaArg, VoidExprArg, voidPtrArg};

      if (Expr* call = m_Sema.ActOnCallExpr(S, checkCall,
                                            Loc, args, Loc).get())
      {
        // It is unclear whether this is the correct cast if the type
        // is dependent.  Hence, For now, we do not expect SynthesizeCheck to
        // be run on a function template.  It should be run only on function
        // instances.
        clang::TypeSourceInfo* argTSI = m_Context.getTrivialTypeSourceInfo(
                                          Arg->getType(), Loc);
        Expr* castExpr = m_Sema.BuildCStyleCastExpr(Loc, argTSI,
                                                    Loc, call).get();
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
        for (const auto &Idx : NonNull->args()) {
          ArgIndexs.set(Idx.getASTIndex());
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
                                        Sema::ForVisibleRedeclaration);
      m_Sema.LookupQualifiedName(*m_clingthrowIfInvalidPointerCache,
                                 m_Context.getTranslationUnitDecl());
      assert(!m_clingthrowIfInvalidPointerCache->empty() &&
              "Lookup of cling_runtime_internal_throwIfInvalidPointer failed!");
    }
  };

  static bool hasPtrCheckDisabledInContext(Sema *S, const Decl* D) {
    if (isa<TranslationUnitDecl>(D))
      return false;
    for (const auto *Ann : D->specific_attrs<clang::AnnotateAttr>()) {
      if (Ann->getAnnotation() == "__cling__ptrcheck(off)")
        return true;
      else if (Ann->getAnnotation() == "__cling__ptrcheck(on)")
        return false;
      else if (Ann->getAnnotation().startswith("__cling__ptrcheck(")) {
        DiagnosticsEngine& Diags = S->getDiagnostics();
        Diags.Report(Ann->getLocation(),
                    Diags.getCustomDiagID(
                        clang::DiagnosticsEngine::Level::Warning,
                        "NullDerefProtectionTransformer: invalid annotation: '%0'. Must be `on` or `off`."))
            << Ann->getAnnotation();
      }
    }
    const Decl* Parent = nullptr;
    for (auto DC = D->getDeclContext(); !Parent; DC = DC->getParent())
      Parent = dyn_cast<Decl>(DC);

    assert(Parent && "Decl without context!");

    return hasPtrCheckDisabledInContext(S, Parent);
  }

} // unnamed namespace

namespace cling {
  NullDerefProtectionTransformer::NullDerefProtectionTransformer(Interpreter* I)
    : ASTTransformer(&I->getCI()->getSema()), m_Interp(I) {
  }

  NullDerefProtectionTransformer::~NullDerefProtectionTransformer()
  { }

  bool NullDerefProtectionTransformer::shouldTransform(const clang::Decl* D) {
    if (D->isFromASTFile())
      return false;
    if (D->isTemplateDecl())
      return false;

    if (hasPtrCheckDisabledInContext(getSemaPtr(), D))
      return false;

    auto Loc = D->getLocation();
    if (Loc.isInvalid())
      return false;

    SourceManager& SM = m_Interp->getSema().getSourceManager();
    auto Characteristic = SM.getFileCharacteristic(Loc);
    if (Characteristic != clang::SrcMgr::C_User)
      return false;

    auto FID = SM.getFileID(Loc);
    if (FID.isInvalid())
      return false;

    auto FE = SM.getFileEntryForID(FID);
    if (!FE)
      return false;

    auto Dir = FE->getDir();
    if (!Dir)
      return false;

    auto IterAndInserted = m_ShouldVisitDir.try_emplace(Dir, true);
    if (IterAndInserted.second == false)
      return IterAndInserted.first->second;

    if (llvm::sys::fs::can_write(Dir->getName()))
      return true; // `true` is already emplaced above.

    // Remember that this dir is not writable and should not be visited.
    IterAndInserted.first->second = false;
    return false;
  }


  ASTTransformer::Result
  NullDerefProtectionTransformer::Transform(clang::Decl* D) {
    if (getCompilationOpts().CheckPointerValidity && shouldTransform(D)) {
      PointerCheckInjector injector(*m_Interp);
      injector.TraverseDecl(D);
    }
    return Result(D, true);
  }
} // end namespace cling
