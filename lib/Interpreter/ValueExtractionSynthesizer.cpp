//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "ValueExtractionSynthesizer.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Interpreter/Value.h"
#include "cling/Utils/AST.h"
#include "cling/Utils/Output.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

using namespace clang;

namespace cling {
  ValueExtractionSynthesizer::ValueExtractionSynthesizer(clang::Sema* S,
                                                         bool isChildInterpreter)
    : WrapperTransformer(S), m_Context(&S->getASTContext()), m_gClingVD(0),
      m_UnresolvedNoAlloc(0), m_UnresolvedWithAlloc(0),
      m_UnresolvedCopyArray(0), m_isChildInterpreter(isChildInterpreter) { }

  // pin the vtable here.
  ValueExtractionSynthesizer::~ValueExtractionSynthesizer() { }

  namespace {
    class ReturnStmtCollector : public StmtVisitor<ReturnStmtCollector> {
    private:
      llvm::SmallVectorImpl<Stmt**>& m_Stmts;
    public:
      ReturnStmtCollector(llvm::SmallVectorImpl<Stmt**>& S)
        : m_Stmts(S) {}

      void VisitStmt(Stmt* S) {
        for(Stmt::child_iterator I = S->child_begin(), E = S->child_end();
            I != E; ++I) {
          if (!*I)
            continue;
          if (isa<LambdaExpr>(*I))
            continue;
          Visit(*I);
          if (isa<ReturnStmt>(*I))
            m_Stmts.push_back(&*I);
        }
      }
    };
  }

  ASTTransformer::Result ValueExtractionSynthesizer::Transform(clang::Decl* D) {
    const CompilationOptions& CO = getCompilationOpts();
    // If we do not evaluate the result, or printing out the result return.
    if (!(CO.ResultEvaluation || CO.ValuePrinting))
      return Result(D, true);

    FunctionDecl* FD = cast<FunctionDecl>(D);
    assert(utils::Analyze::IsWrapper(FD) && "Expected wrapper");

    int foundAtPos = -1;
    Expr* lastExpr = utils::Analyze::GetOrCreateLastExpr(FD, &foundAtPos,
                                                         /*omitDS*/false,
                                                         m_Sema);
    if (foundAtPos < 0)
      return Result(D, true);

    typedef llvm::SmallVector<Stmt**, 4> StmtIters;
    StmtIters returnStmts;
    ReturnStmtCollector collector(returnStmts);
    CompoundStmt* CS = cast<CompoundStmt>(FD->getBody());
    collector.VisitStmt(CS);

    if (isa<Expr>(*(CS->body_begin() + foundAtPos)))
      returnStmts.push_back(CS->body_begin() + foundAtPos);

    // We want to support cases such as:
    // gCling->evaluate("if() return 'A' else return 12", V), that puts in V,
    // either A or 12.
    // In this case the void wrapper is compiled with the stmts returning
    // values. Sema would cast them to void, but the code will still be
    // executed. For example:
    // int g(); void f () { return g(); } will still call g().
    //
    for (StmtIters::iterator I = returnStmts.begin(), E = returnStmts.end();
         I != E; ++I) {
      ReturnStmt* RS = dyn_cast<ReturnStmt>(**I);
      if (RS) {
        // When we are handling a return stmt, the last expression must be the
        // return stmt value. Ignore the calculation of the lastStmt because it
        // might be wrong, in cases where the return is not in the end of the
        // function.
        lastExpr = RS->getRetValue();
        if (lastExpr) {
          assert (lastExpr->getType()->isVoidType() && "Must be void type.");
          // Any return statement will have been "healed" by Sema
          // to correspond to the original void return type of the
          // wrapper, using a ImplicitCastExpr 'void' <ToVoid>.
          // Remove that.
          if (ImplicitCastExpr* VoidCast
              = dyn_cast<ImplicitCastExpr>(lastExpr)) {
            lastExpr = VoidCast->getSubExpr();
          }
        }
        // if no value assume void
        else {
          // We can't PushDeclContext, because we don't have scope.
          Sema::ContextRAII pushedDC(*m_Sema, FD);
          RS->setRetValue(SynthesizeSVRInit(0));
        }

      }
      else
        lastExpr = cast<Expr>(**I);

      if (lastExpr) {
        QualType lastExprTy = lastExpr->getType();
        // May happen on auto types which resolve to dependent.
        if (lastExprTy->isDependentType())
          continue;
        // Set up lastExpr properly.
        // Change the void function's return type
        // We can't PushDeclContext, because we don't have scope.
        Sema::ContextRAII pushedDC(*m_Sema, FD);

        if (lastExprTy->isFunctionType()) {
          // A return type of function needs to be converted to
          // pointer to function.
          lastExprTy = m_Context->getPointerType(lastExprTy);
          lastExpr = m_Sema->ImpCastExprToType(lastExpr, lastExprTy,
                                               CK_FunctionToPointerDecay,
                                               VK_RValue).get();
        }

        //
        // Here we don't want to depend on the JIT runFunction, because of its
        // limitations, when it comes to return value handling. There it is
        // not clear who provides the storage and who cleans it up in a
        // platform independent way.
        //
        // Depending on the type we need to synthesize a call to cling:
        // 0) void : set the value's type to void;
        // 1) enum, integral, float, double, referece, pointer types :
        //      call to cling::internal::setValueNoAlloc(...);
        // 2) object type (alloc on the stack) :
        //      cling::internal::setValueWithAlloc
        //   2.1) constant arrays:
        //          call to cling::runtime::internal::copyArray(...)
        //
        // We need to synthesize later:
        // Wrapper has signature: void w(cling::Value SVR)
        // case 1):
        //   setValueNoAlloc(gCling, &SVR, lastExprTy, lastExpr())
        // case 2):
        //   new (setValueWithAlloc(gCling, &SVR, lastExprTy)) (lastExpr)
        // case 2.1):
        //   copyArray(src, placement, size)

        Expr* SVRInit = SynthesizeSVRInit(lastExpr);
        // if we had return stmt update to execute the SVR init, even if the
        // wrapper returns void.
        if (SVRInit) {
          if (RS) {
            if (ImplicitCastExpr* VoidCast
                = dyn_cast<ImplicitCastExpr>(RS->getRetValue()))
              VoidCast->setSubExpr(SVRInit);
          } else
            **I = SVRInit;
        } else {
          // FIXME: Do this atomically or something so that AST context will not
          // contain Expr(s) that are unused for the rest of it's life.
          return Result(D, false);
        }
      }
    }
    return Result(D, true);
  }

// Helper function for the SynthesizeSVRInit
namespace {
  static bool availableCopyConstructor(QualType QT, clang::Sema* S) {
    // Check the the existance of the copy constructor the tha placement new will use.
    if (CXXRecordDecl* RD = QT->getAsCXXRecordDecl()) {
      // If it has a trivial copy constructor it is accessible and it is callable.
      if(RD->hasTrivialCopyConstructor()) return true;
      // Lookup the copy canstructor and check its accessiblity.
      if (CXXConstructorDecl* CD = S->LookupCopyingConstructor(RD, QT.getCVRQualifiers())) {
        if (!CD->isDeleted() && CD ->getAccess() == clang::AccessSpecifier::AS_public) {
          return true;
        }
      }
      return false;
    }
    return true;
  }
}

  Expr* ValueExtractionSynthesizer::SynthesizeSVRInit(Expr* E) {
    if (!m_gClingVD && !FindAndCacheRuntimeDecls())
      return nullptr;

    // Build a reference to gCling
    ExprResult gClingDRE
      = m_Sema->BuildDeclRefExpr(m_gClingVD, m_Context->VoidPtrTy,
                                 VK_RValue, SourceLocation());
    // We have the wrapper as Sema's CurContext
    FunctionDecl* FD = cast<FunctionDecl>(m_Sema->CurContext);

    ExprWithCleanups* Cleanups = 0;
    // In case of ExprWithCleanups we need to extend its 'scope' to the call.
    if (E && isa<ExprWithCleanups>(E)) {
      Cleanups = cast<ExprWithCleanups>(E);
      E = Cleanups->getSubExpr();
    }

    // Build a reference to Value* in the wrapper, should be
    // the only argument of the wrapper.
    SourceLocation locStart = (E) ? E->getLocStart() : FD->getLocStart();
    SourceLocation locEnd = (E) ? E->getLocEnd() : FD->getLocEnd();
    ExprResult wrapperSVRDRE
      = m_Sema->BuildDeclRefExpr(FD->getParamDecl(0), m_Context->VoidPtrTy,
                                 VK_RValue, locStart);
    QualType ETy = (E) ? E->getType() : m_Context->VoidTy;
    QualType desugaredTy = ETy.getDesugaredType(*m_Context);

    // The expr result is transported as reference, pointer, array, float etc
    // based on the desugared type. We should still expose the typedef'ed
    // (sugared) type to the cling::Value.
    if (desugaredTy->isRecordType() && E->getValueKind() == VK_LValue) {
      // returning a lvalue (not a temporary): the value should contain
      // a reference to the lvalue instead of copying it.
      desugaredTy = m_Context->getLValueReferenceType(desugaredTy);
      ETy = m_Context->getLValueReferenceType(ETy);
    }
    Expr* ETyVP
      = utils::Synthesize::CStyleCastPtrExpr(m_Sema, m_Context->VoidPtrTy,
                                             (uintptr_t)ETy.getAsOpaquePtr());

    // Pass whether to Value::dump() or not:
    Expr* EVPOn
      = new (*m_Context) CharacterLiteral(getCompilationOpts().ValuePrinting,
                                          CharacterLiteral::Ascii,
                                          m_Context->CharTy,
                                          SourceLocation());

    llvm::SmallVector<Expr*, 6> CallArgs;
    CallArgs.push_back(gClingDRE.get());
    CallArgs.push_back(wrapperSVRDRE.get());
    CallArgs.push_back(ETyVP);
    CallArgs.push_back(EVPOn);

    ExprResult Call;
    SourceLocation noLoc = locStart;
    if (desugaredTy->isVoidType()) {
      // In cases where the cling::Value gets reused we need to reset the
      // previous settings to void.
      // We need to synthesize setValueNoAlloc(...), E, because we still need
      // to run E.

      // FIXME: Suboptimal: this discards the already created AST nodes.
      QualType vpQT = m_Context->VoidPtrTy;
      QualType vQT = m_Context->VoidTy;
      Expr* vpQTVP
        = utils::Synthesize::CStyleCastPtrExpr(m_Sema, vpQT,
                                               (uintptr_t)vQT.getAsOpaquePtr());
      CallArgs[2] = vpQTVP;


      Call = m_Sema->ActOnCallExpr(/*Scope*/0, m_UnresolvedNoAlloc,
                                   locStart, CallArgs, locEnd);

      if (E)
        Call = m_Sema->CreateBuiltinBinOp(locStart, BO_Comma, Call.get(), E);

    }
    else if (desugaredTy->isRecordType() || desugaredTy->isConstantArrayType()
             || desugaredTy->isMemberPointerType()) {
      // 2) object types :
      // check existence of copy constructor before call
      if (!desugaredTy->isMemberPointerType()
          && !availableCopyConstructor(desugaredTy, m_Sema))
        return E;
      // call new (setValueWithAlloc(gCling, &SVR, ETy)) (E)
      Call = m_Sema->ActOnCallExpr(/*Scope*/0, m_UnresolvedWithAlloc,
                                   locStart, CallArgs, locEnd);
      Expr* placement = Call.get();
      if (const ConstantArrayType* constArray
          = dyn_cast<ConstantArrayType>(desugaredTy.getTypePtr())) {
        CallArgs.clear();
        CallArgs.push_back(E);
        CallArgs.push_back(placement);
        size_t arrSize
          = m_Context->getConstantArrayElementCount(constArray);
        Expr* arrSizeExpr
          = utils::Synthesize::IntegerLiteralExpr(*m_Context, arrSize);

        CallArgs.push_back(arrSizeExpr);
        // 2.1) arrays:
        // call copyArray(T* src, void* placement, size_t size)
        Call = m_Sema->ActOnCallExpr(/*Scope*/0, m_UnresolvedCopyArray,
                                     locStart, CallArgs, locEnd);

      }
      else {
        if (!E->getSourceRange().isValid()) {
          // We cannot do CXXNewExpr::CallInit (see Sema::BuildCXXNew) but
          // that's what we want. Fail...
          return E;
        }
        TypeSourceInfo* ETSI
          = m_Context->getTrivialTypeSourceInfo(ETy, noLoc);

        assert(!Call.isInvalid() && "Invalid Call before building new");

        Call = m_Sema->BuildCXXNew(E->getSourceRange(),
                                   /*useGlobal ::*/true,
                                   /*placementLParen*/ noLoc,
                                   MultiExprArg(placement),
                                   /*placementRParen*/ noLoc,
                                   /*TypeIdParens*/ SourceRange(),
                                   /*allocType*/ ETSI->getType(),
                                   /*allocTypeInfo*/ETSI,
                                   /*arraySize*/0,
                                   /*directInitRange*/E->getSourceRange(),
                                   /*initializer*/E
                                   );
        if (Call.isInvalid()) {
          m_Sema->Diag(E->getLocStart(), diag::err_undeclared_var_use)
            << "operator new";
          return Call.get();
        }

        // Handle possible cleanups:
        Call = m_Sema->ActOnFinishFullExpr(Call.get());
      }
    }
    else {
      // Mark the current number of arguemnts
      const size_t nArgs = CallArgs.size();
      if (desugaredTy->isIntegralOrEnumerationType()) {
        // 1)  enum, integral, float, double, referece, pointer types :
        //      call to cling::internal::setValueNoAlloc(...);

        // force-cast it into uint64 in order to pick up the correct overload.
        QualType UInt64Ty = m_Context->UnsignedLongLongTy;
        TypeSourceInfo* TSI
          = m_Context->getTrivialTypeSourceInfo(UInt64Ty, noLoc);
        Expr* castedE
          = m_Sema->BuildCStyleCastExpr(noLoc, TSI, noLoc, E).get();
        CallArgs.push_back(castedE);
      }
      else if (desugaredTy->isReferenceType()) {
        // we need to get the address of the references
        Expr* AddrOfE  = m_Sema->CreateBuiltinUnaryOp(noLoc, UO_AddrOf, E).get();
        CallArgs.push_back(AddrOfE);
      }
      else if (desugaredTy->isAnyPointerType()) {
        // function pointers need explicit void* cast.
        QualType VoidPtrTy = m_Context->VoidPtrTy;
        TypeSourceInfo* TSI
          = m_Context->getTrivialTypeSourceInfo(VoidPtrTy, noLoc);
        Expr* castedE
          = m_Sema->BuildCStyleCastExpr(noLoc, TSI, noLoc, E).get();
        CallArgs.push_back(castedE);
      }
      else if (desugaredTy->isNullPtrType()) {
        // nullptr should decay to void* just fine.
        CallArgs.push_back(E);
      }
      else if (desugaredTy->isFloatingType()) {
        // floats and double will fall naturally in the correct
        // case, because of the overload resolution.
        CallArgs.push_back(E);
      }

      // Test CallArgs.size to make sure an additional argument (the value)
      // has been pushed on, if not than we didn't know how to handle the type
      if (CallArgs.size() > nArgs) {
        Call = m_Sema->ActOnCallExpr(/*Scope*/0, m_UnresolvedNoAlloc,
                                   locStart, CallArgs, locEnd);
      }
      else {
        m_Sema->Diag(locStart, diag::err_unsupported_unknown_any_decl) <<
          utils::TypeName::GetFullyQualifiedName(desugaredTy, *m_Context) <<
          SourceRange(locStart, locEnd);
      }
    }

    assert(!Call.isInvalid() && "Invalid Call");

    // Extend the scope of the temporary cleaner if applicable.
    if (Cleanups && !Call.isInvalid()) {
      Cleanups->setSubExpr(Call.get());
      Cleanups->setValueKind(Call.get()->getValueKind());
      Cleanups->setType(Call.get()->getType());
      return Cleanups;
    }
    return Call.get();
  }

  static bool VSError(const char* err) {
    cling::errs() << "ValueExtractionSynthesizer error: " << err << ".\n";
    return false;
  }

  bool ValueExtractionSynthesizer::FindAndCacheRuntimeDecls() {
    assert(!m_gClingVD && "Called multiple times!?");
    DeclContext* NSD = m_Context->getTranslationUnitDecl();
    clang::VarDecl* clingVD = nullptr;
    if (m_Sema->getLangOpts().CPlusPlus) {
      if (!(NSD = utils::Lookup::Namespace(m_Sema, "cling")))
        return VSError("cling namespace not defined");
      if (!(NSD = utils::Lookup::Namespace(m_Sema, "runtime", NSD)))
        return VSError("cling::runtime namespace not defined");
      if (!(clingVD = cast<VarDecl>(utils::Lookup::Named(m_Sema, "gCling",
                                                            NSD))))
        return VSError("cling::runtime::gCling not defined");
      if (!NSD)
        return VSError("cling::runtime namespace not defined");

      if (!(NSD = utils::Lookup::Namespace(m_Sema, "internal", NSD)))
        return VSError("cling::runtime::internal namespace not defined");
    }
    LookupResult R(*m_Sema, &m_Context->Idents.get("setValueNoAlloc"),
                   SourceLocation(), Sema::LookupOrdinaryName,
                   Sema::ForRedeclaration);

    m_Sema->LookupQualifiedName(R, NSD);
    if (R.empty())
      return VSError("Cannot find cling::runtime::internal::setValueNoAlloc");

    const bool ADL = false;
    CXXScopeSpec CSS;
    m_UnresolvedNoAlloc = m_Sema->BuildDeclarationNameExpr(CSS, R, ADL).get();
    if (!m_UnresolvedNoAlloc)
      return VSError("Could not build cling::runtime::internal"
                     "::setValueNoAlloc");

    R.clear();
    R.setLookupName(&m_Context->Idents.get("setValueWithAlloc"));
    m_Sema->LookupQualifiedName(R, NSD);
    if (R.empty())
      return VSError("Cannot find cling::runtime::internal::setValueWithAlloc");

    m_UnresolvedWithAlloc = m_Sema->BuildDeclarationNameExpr(CSS, R, ADL).get();
    if (!m_UnresolvedWithAlloc)
      return VSError("Could not build cling::runtime::internal"
                     "::setValueWithAlloc");

    R.clear();
    R.setLookupName(&m_Context->Idents.get("copyArray"));
    m_Sema->LookupQualifiedName(R, NSD);
    // FIXME: In the case of the multiple interpreters (parent-child),
    // the child interpreter doesn't include the runtime universe.
    // The child interpreter will try to import this function from its
    // parent interpreter, but it will fail, because this is a template function.
    // Once the import of template functions becomes supported by clang,
    // this check can be de-activated.
    if (!m_isChildInterpreter && R.empty())
      return VSError("Cannot find cling::runtime::internal::copyArray");

    m_UnresolvedCopyArray = m_Sema->BuildDeclarationNameExpr(CSS, R, ADL).get();
    if (!m_UnresolvedCopyArray)
      return VSError("Could not build cling::runtime::internal::copyArray");

    m_gClingVD = clingVD;
    return true;
  }
} // end namespace cling


// Provide implementation of the functions that ValueExtractionSynthesizer calls
namespace {

  static void dumpIfNoStorage(void* vpV, char vpOn) {
    const cling::Value& V = *(cling::Value*)vpV;
    // If the value copies over the temporary we must delay the printing until
    // the temporary gets copied over. For the rest of the temporaries we *must*
    // dump here because their lifetime will be gone otherwise. Eg.
    //
    // std::string f(); f().c_str() // have to dump during the same stmt.
    //
    assert(!V.needsManagedAllocation() && "Must contain non managed temporary");
    assert(vpOn != (char)cling::CompilationOptions::VPAuto
           && "VPAuto must have been expanded earlier.");
    if (vpOn == (char)cling::CompilationOptions::VPEnabled)
      V.dump();
  }

  ///\brief Allocate the Value and return the Value
  /// for an expression evaluated at the prompt.
  ///
  ///\param [in] interp - The cling::Interpreter to allocate the SToredValueRef.
  ///\param [in] vpQT - The opaque ptr for the clang::QualType of value stored.
  ///\param [out] vpStoredValRef - The Value that is allocated.
  static cling::Value&
  allocateStoredRefValueAndGetGV(void* vpI, void* vpSVR, void* vpQT) {
    cling::Interpreter* i = (cling::Interpreter*)vpI;
    clang::QualType QT = clang::QualType::getFromOpaquePtr(vpQT);
    cling::Value& SVR = *(cling::Value*)vpSVR;
    // Here the copy keeps the refcounted value alive.
    SVR = cling::Value(QT, *i);
    return SVR;
  }
}
namespace cling {
namespace runtime {
  namespace internal {
    void setValueNoAlloc(void* vpI, void* vpSVR, void* vpQT, char vpOn) {
      // In cases of void we 'just' need to change the type of the value.
      allocateStoredRefValueAndGetGV(vpI, vpSVR, vpQT);
    }
    void setValueNoAlloc(void* vpI, void* vpSVR, void* vpQT, char vpOn,
                         float value) {
      allocateStoredRefValueAndGetGV(vpI, vpSVR, vpQT).getAs<float>() = value;
      dumpIfNoStorage(vpSVR, vpOn);
    }
    void setValueNoAlloc(void* vpI, void* vpSVR, void* vpQT, char vpOn,
                         double value) {
      allocateStoredRefValueAndGetGV(vpI, vpSVR, vpQT).getAs<double>() = value;
      dumpIfNoStorage(vpSVR, vpOn);
    }
    void setValueNoAlloc(void* vpI, void* vpSVR, void* vpQT, char vpOn,
                         long double value) {
      allocateStoredRefValueAndGetGV(vpI, vpSVR, vpQT).getAs<long double>()
        = value;
      dumpIfNoStorage(vpSVR, vpOn);
    }
    void setValueNoAlloc(void* vpI, void* vpSVR, void* vpQT, char vpOn,
                         unsigned long long value) {
      allocateStoredRefValueAndGetGV(vpI, vpSVR, vpQT)
        .getAs<unsigned long long>() = value;
      dumpIfNoStorage(vpSVR, vpOn);
    }
    void setValueNoAlloc(void* vpI, void* vpSVR, void* vpQT, char vpOn,
                         const void* value){
      allocateStoredRefValueAndGetGV(vpI, vpSVR, vpQT).getAs<void*>()
        = const_cast<void*>(value);
      dumpIfNoStorage(vpSVR, vpOn);
    }
    void* setValueWithAlloc(void* vpI, void* vpSVR, void* vpQT, char vpOn) {
      return allocateStoredRefValueAndGetGV(vpI, vpSVR, vpQT).getAs<void*>();
    }
  } // end namespace internal
} // end namespace runtime
} // end namespace cling
