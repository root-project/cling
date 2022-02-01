//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "DeclExtractor.h"

#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

using namespace clang;

namespace {
  // Remove the linkage cache. On next access it will calculate it
  // considering the new position of the declaration.
  class BreakProtection: public clang::NamedDecl {
  public:
    static void resetCachedLinkage(clang::NamedDecl* ND) {
      static_cast<BreakProtection*>(ND)->CacheValidAndLinkage = 0; }
  };

  static void clearLinkageForClass(const CXXRecordDecl *RD) {
    for (DeclContext::decl_iterator I = RD->decls_begin(),
           E = RD->decls_end(); I != E; ++I) {
      if (NamedDecl* IND = dyn_cast<NamedDecl>(*I))
        BreakProtection::resetCachedLinkage(IND);
    }
  }

  static void clearLinkage(NamedDecl *ND) {
    BreakProtection::resetCachedLinkage(ND);
    if (const CXXRecordDecl* CXXRD = dyn_cast<CXXRecordDecl>(ND))
      clearLinkageForClass(CXXRD);
    else if (ClassTemplateDecl *CTD = dyn_cast<ClassTemplateDecl>(ND)) {
      // Clear linkage for the template pattern.
      CXXRecordDecl *record = CTD->getTemplatedDecl();
      clearLinkageForClass(record);

      // We need to clear linkage for specializations, too.
      for (ClassTemplateDecl::spec_iterator
             i = CTD->spec_begin(), e = CTD->spec_end(); i != e; ++i)
        clearLinkage(*i);
    } else if (FunctionTemplateDecl *FTD= dyn_cast<FunctionTemplateDecl>(ND)) {
      // Clear cached linkage for function template decls, too.
      clearLinkage(FTD->getTemplatedDecl());
      for (FunctionTemplateDecl::spec_iterator
             i = FTD->spec_begin(), e = FTD->spec_end(); i != e; ++i)
        clearLinkage(*i);
    }
  }
}

namespace cling {

  DeclExtractor::DeclExtractor(Sema* S)
    : WrapperTransformer(S), m_Context(&S->getASTContext()),
      m_UniqueNameCounter(0)
  { }

  // pin the vtable here
  DeclExtractor::~DeclExtractor()
  { }

  WrapperTransformer::Result DeclExtractor::Transform(Decl* D) {
    if (!getCompilationOpts().DeclarationExtraction)
      return Result(D, true);
    FunctionDecl* FD = cast<FunctionDecl>(D);
    assert(utils::Analyze::IsWrapper(FD) && "Expected wrapper");

    if (!ExtractDecl(FD))
      return Result(nullptr, false);
    return Result(FD, true);
  }

  bool DeclExtractor::ValidateCXXRecord(VarDecl* VD) const {
    QualType QT = VD->getType();
    if (CXXRecordDecl* RD = QT->getAsCXXRecordDecl()) {
      if (RD->isLambda()) {
        bool Copy = RD->getLambdaCaptureDefault() == LCD_ByCopy;
        if (!Copy) {
          for (auto&& Capture : RD->captures()) {
            if ((Copy = Capture.getCaptureKind() == LCK_ByCopy)) break;
          }
        }
        if (Copy) {
          const int ID = m_Context->getDiagnostics().getCustomDiagID(
              DiagnosticsEngine::Warning,
              "captures will be by reference, not copy");
          m_Context->getDiagnostics().Report(VD->getSourceRange().getBegin(),
                                             ID);
          // Warning is good enough, no reason to fail over this
          return true;
        }
      }
    }
    return true;
  }

  bool DeclExtractor::ExtractDecl(FunctionDecl* FD) {
    llvm::SmallVector<NamedDecl*, 4> TouchedDecls;
    CompoundStmt* CS = dyn_cast<CompoundStmt>(FD->getBody());
    assert(CS && "Function body not a CompoundStmt?");
    assert(utils::Analyze::IsWrapper(FD) && "FD not a Cling wrapper?");
    // DC is the internal `__cling_N5xxx' namespace or (if decl shadowing if off), the TU
    DeclContext* WrapperDC = FD->getDeclContext();
    Scope* TUScope = m_Sema->TUScope;
    llvm::SmallVector<Stmt*, 4> Stmts;

    if (CS->body_empty())
      return FD;

    for (CompoundStmt::body_iterator I = CS->body_begin(), EI = CS->body_end();
         I != EI; ++I) {
      DeclStmt* DS = dyn_cast<DeclStmt>(*I);
      if (!DS) {
        Stmts.push_back(*I);
        continue;
      }

      for (DeclStmt::decl_iterator J = DS->decl_begin();
           J != DS->decl_end(); ++J) {
        NamedDecl* ND = dyn_cast<NamedDecl>(*J);
        if (isa<UsingDirectiveDecl>(*J))
          continue; // FIXME: Here we should be more elegant.
        if (ND) {
          if (Stmts.size()) {
            // We need to emit a new custom wrapper wrapping the stmts
            EnforceInitOrder(Stmts);
            assert(!Stmts.size() && "Stmt list must be flushed.");
          }

          DeclContext* OldDC = ND->getDeclContext();

          // Make sure the decl is not found at its old possition
          ND->getLexicalDeclContext()->removeDecl(ND);
          if (Scope* S = m_Sema->getScopeForContext(OldDC)) {
            S->RemoveDecl(ND);
            if (utils::Analyze::isOnScopeChains(ND, *m_Sema))
              m_Sema->IdResolver.RemoveDecl(ND);
          }

          // For variable definitions causing var/function ambiguity such as:
          // MyClass my();, C++ standard says it shall be resolved as a function
          //
          // In the particular context this definition is inside a function
          // already, but clang thinks it as a lambda, so we need to ignore the
          // check decl context vs lexical decl context.
          DeclContext *NewDC = isa<TagDecl>(ND) ? m_Context->getTranslationUnitDecl()
		                                : WrapperDC;
          if (ND->getDeclContext() == ND->getLexicalDeclContext()
              || isa<FunctionDecl>(ND))
            ND->setLexicalDeclContext(NewDC);
          else
            assert(0 && "Not implemented: Decl with different lexical context");
          ND->setDeclContext(NewDC);

          if (VarDecl* VD = dyn_cast<VarDecl>(ND)) {
            if (!ValidateCXXRecord(VD))
              return false;
            VD->setStorageClass(SC_None);
          }

          clearLinkage(ND);

          TouchedDecls.push_back(ND);

          Emit(DeclGroupRef(ND));
        }
      }
    }
    bool hasNoErrors = !CheckForClashingNames(TouchedDecls, WrapperDC);
    if (hasNoErrors) {
      for (size_t i = 0; i < TouchedDecls.size(); ++i) {
        // The transparent DeclContexts (eg. scopeless enum) doesn't have
        // scopes. While extracting their contents we need to update the
        // lookup tables and telling them to pick up the new positions
        // in the AST.
        if (DeclContext* InnerDC = dyn_cast<DeclContext>(TouchedDecls[i])) {
          if (InnerDC->isTransparentContext()) {
            // We can't PushDeclContext, because we don't have scope.
            Sema::ContextRAII pushedDC(*m_Sema, InnerDC);

            for(DeclContext::decl_iterator DI = InnerDC->decls_begin(),
                  DE = InnerDC->decls_end(); DI != DE ; ++DI) {
              if (NamedDecl* ND = dyn_cast<NamedDecl>(*DI))
                InnerDC->makeDeclVisibleInContext(ND);
            }
          }
        }

        // We should skip the checks for anonymous decls and we should not
        // register them in the lookup. Their inner decls have been added above.
        if (!TouchedDecls[i]->getDeclName())
          continue;

        Sema::ContextRAII RAII(*m_Sema, TouchedDecls[i]->getDeclContext());
        m_Sema->PushOnScopeChains(TouchedDecls[i],
                                  TUScope,
                    /*AddCurContext*/!isa<UsingDirectiveDecl>(TouchedDecls[i]));
      }
    }

    // Create a new body.
    auto newCS = CompoundStmt::Create(*m_Context, Stmts, CS->getLBracLoc(),
                                      CS->getRBracLoc());
    FD->setBody(newCS);

    if (hasNoErrors && !TouchedDecls.empty()) {
      // Put the wrapper after its declarations. (Nice when AST dumping)
      WrapperDC->removeDecl(FD);
      WrapperDC->addDecl(FD);
    }

    return hasNoErrors ? FD : 0;
  }

  void DeclExtractor::createUniqueName(std::string& out) {
    if (out.empty())
      out += '_';

    llvm::raw_string_ostream(out) << "_init_order"
      << utils::Synthesize::UniquePrefix << m_UniqueNameCounter++;
  }

  void DeclExtractor::EnforceInitOrder(llvm::SmallVectorImpl<Stmt*>& Stmts){
    Scope* TUScope = m_Sema->TUScope;
    DeclContext* TUDC = static_cast<DeclContext*>(TUScope->getEntity());
    // We can't PushDeclContext, because we don't have scope.
    Sema::ContextRAII pushedDC(*m_Sema, TUDC);

    std::string FunctionName = "__fd";
    createUniqueName(FunctionName);
    IdentifierInfo& IIFD = m_Context->Idents.get(FunctionName);
    SourceLocation Loc;
    NamedDecl* ND = m_Sema->ImplicitlyDefineFunction(Loc, IIFD, TUScope);
    if (FunctionDecl* FD = dyn_cast_or_null<FunctionDecl>(ND)) {
      Sema::SynthesizedFunctionScope Scope(*m_Sema, FD);
      FD->setImplicit(false); // Better for debugging

      // Add a return statement if it doesn't exist
      if (!isa<ReturnStmt>(Stmts.back())) {
        Sema::ContextRAII pushedDC(*m_Sema, FD);
        // Generate the return statement:
        // First a literal 0, then the return taking that literal.
        // One bit is enough:
        llvm::APInt ZeroInt(m_Context->getIntWidth(m_Context->IntTy), 0,
                            /*isSigned=*/true);
        IntegerLiteral* ZeroLit
          = IntegerLiteral::Create(*m_Context, ZeroInt, m_Context->IntTy,
                                   SourceLocation());
        Stmts.push_back(m_Sema->ActOnReturnStmt(ZeroLit->getExprLoc(),
                                                ZeroLit,
                                                m_Sema->getCurScope()).get());
      }

      // Wrap Stmts into a function body.
      llvm::ArrayRef<Stmt*> StmtsRef(Stmts.data(), Stmts.size());
      CompoundStmt* CS = CompoundStmt::Create(*m_Context, StmtsRef, Loc, Loc);
      FD->setBody(CS);
      Emit(FD);

      // Create the VarDecl with the init
      std::string VarName = "__vd";
      createUniqueName(VarName);
      IdentifierInfo& IIVD = m_Context->Idents.get(VarName);
      VarDecl* VD = VarDecl::Create(*m_Context, TUDC, Loc, Loc, &IIVD,
                                    FD->getReturnType(), (TypeSourceInfo*)0,
                                    SC_None);
      LookupResult R(*m_Sema, FD->getDeclName(), Loc, Sema::LookupMemberName);
      R.addDecl(FD);
      CXXScopeSpec CSS;
      Expr* UnresolvedLookup
        = m_Sema->BuildDeclarationNameExpr(CSS, R, /*ADL*/ false).get();
      Expr* TheCall = m_Sema->ActOnCallExpr(TUScope, UnresolvedLookup, Loc,
                                            MultiExprArg(), Loc).get();
      assert(VD && TheCall && "Missing VD or its init!");
      VD->setInit(TheCall);

      Emit(VD); // Add it to the transaction for codegenning
      TUDC->addHiddenDecl(VD);
      Stmts.clear();
      return;
    }
    llvm_unreachable("Must be able to enforce init order.");
  }

  ///\brief Checks for clashing names when trying to extract a declaration.
  ///
  ///\returns true if there is another declaration with the same name
  bool DeclExtractor::CheckForClashingNames(
                                  const llvm::SmallVector<NamedDecl*, 4>& Decls,
                                            DeclContext* DC) {
    for (size_t i = 0; i < Decls.size(); ++i) {
      NamedDecl* ND = Decls[i];

      if (TagDecl* TD = dyn_cast<TagDecl>(ND)) {
        LookupResult Previous(*m_Sema, ND->getDeclName(), ND->getLocation(),
                              Sema::LookupTagName, Sema::ForVisibleRedeclaration
                              );

        m_Sema->LookupQualifiedName(Previous, DC);

        // There is no function diagnosing the redeclaration of tags (eg. enums).
        // So either we have to do it by hand or we can call the top-most
        // function that does the check. Currently the top-most clang function
        // doing the checks creates an AST node, which we don't want.
        if (!CheckTagDeclaration(TD, Previous))
          return true;
      }
      else if (VarDecl* VD = dyn_cast<VarDecl>(ND)) {
        LookupResult Previous(*m_Sema, ND->getDeclName(), ND->getLocation(),
                              Sema::LookupOrdinaryName, Sema::ForVisibleRedeclaration
                              );
        m_Sema->LookupQualifiedName(Previous, DC);
        m_Sema->CheckVariableDeclaration(VD, Previous);
        if (VD->isInvalidDecl())
          return true;
        // This var decl will likely get referenced later; claim that it's used.
        VD->setIsUsed();
      }
    }

    return false;
  }

  bool DeclExtractor::CheckTagDeclaration(TagDecl* NewTD,
                                          LookupResult& Previous){
    // If the decl is already known invalid, don't check it.
    if (NewTD->isInvalidDecl())
      return false;

    IdentifierInfo* Name = NewTD->getIdentifier();
    // If this is not a definition, it must have a name.
    assert((Name != 0 || NewTD->isThisDeclarationADefinition()) &&
           "Nameless record must be a definition!");

    // Figure out the underlying type if this a enum declaration. We need to do
    // this early, because it's needed to detect if this is an incompatible
    // redeclaration.

    TagDecl::TagKind Kind = NewTD->getTagKind();
    bool Invalid = false;
    assert(NewTD->getNumTemplateParameterLists() == 0
           && "Cannot handle that yet!");
    bool isExplicitSpecialization = false;

    if (Kind == TTK_Enum) {
      EnumDecl* ED = cast<EnumDecl>(NewTD);
      bool ScopedEnum = ED->isScoped();
      const QualType QT = ED->getIntegerType();

      if (QT.isNull() && ScopedEnum)
        // No underlying type explicitly specified, or we failed to parse the
        // type, default to int.
        ; //EnumUnderlying = m_Context->IntTy.getTypePtr();
      else if (!QT.isNull()) {
        // C++0x 7.2p2: The type-specifier-seq of an enum-base shall name an
        // integral type; any cv-qualification is ignored.

        SourceLocation UnderlyingLoc;
        TypeSourceInfo* TI = 0;
        if ((TI = ED->getIntegerTypeSourceInfo()))
          UnderlyingLoc = TI->getTypeLoc().getBeginLoc();

        if (!QT->isDependentType() && !QT->isIntegralType(*m_Context)) {
          m_Sema->Diag(UnderlyingLoc, diag::err_enum_invalid_underlying)
            << QT;
        }
        if (TI)
          m_Sema->DiagnoseUnexpandedParameterPack(UnderlyingLoc, TI,
                                                Sema::UPPC_FixedUnderlyingType);
      }
    }

    DeclContext *SearchDC = m_Sema->CurContext;
    DeclContext *DC = m_Sema->CurContext;
    //bool isStdBadAlloc = false;
    SourceLocation NameLoc = NewTD->getLocation();
    // if (Name && SS.isNotEmpty()) {
    //   // We have a nested-name tag ('struct foo::bar').

    //   // Check for invalid 'foo::'.
    //   if (SS.isInvalid()) {
    //     Name = 0;
    //     goto CreateNewDecl;
    //   }

    //   // If this is a friend or a reference to a class in a dependent
    //   // context, don't try to make a decl for it.
    //   if (TUK == TUK_Friend || TUK == TUK_Reference) {
    //     DC = computeDeclContext(SS, false);
    //     if (!DC) {
    //       IsDependent = true;
    //       return 0;
    //     }
    //   } else {
    //     DC = computeDeclContext(SS, true);
    //     if (!DC) {
    //       Diag(SS.getRange().getBegin(),
    //            diag::err_dependent_nested_name_spec)
    //         << SS.getRange();
    //       return 0;
    //     }
    //   }

    //   if (RequireCompleteDeclContext(SS, DC))
    //     return 0;

    //   SearchDC = DC;
    //   // Look-up name inside 'foo::'.
    //   LookupQualifiedName(Previous, DC);

    //   if (Previous.isAmbiguous())
    //     return 0;

    //   if (Previous.empty()) {
    //     // Name lookup did not find anything. However, if the
    //     // nested-name-specifier refers to the current instantiation,
    //     // and that current instantiation has any dependent base
    //     // classes, we might find something at instantiation time: treat
    //     // this as a dependent elaborated-type-specifier.
    //     // But this only makes any sense for reference-like lookups.
    //     if (Previous.wasNotFoundInCurrentInstantiation() &&
    //         (TUK == TUK_Reference || TUK == TUK_Friend)) {
    //       IsDependent = true;
    //       return 0;
    //     }

    //     // A tag 'foo::bar' must already exist.
    //     Diag(NameLoc, diag::err_not_tag_in_scope)
    //       << Kind << Name << DC << SS.getRange();
    //     Name = 0;
    //     Invalid = true;
    //   goto CreateNewDecl;
    // }
    //} else
    if (Name) {
      // If this is a named struct, check to see if there was a previous forward
      // declaration or definition.
      // FIXME: We're looking into outer scopes here, even when we
      // shouldn't be. Doing so can result in ambiguities that we
      // shouldn't be diagnosing.

      //LookupName(Previous, S);

      if (Previous.isAmbiguous()) {
        LookupResult::Filter F = Previous.makeFilter();
        while (F.hasNext()) {
          NamedDecl *ND = F.next();
          if (ND->getDeclContext()->getRedeclContext() != SearchDC)
            F.erase();
        }
        F.done();
      }

      // Note:  there used to be some attempt at recovery here.
      if (Previous.isAmbiguous()) {
        return false;
      }

      if (!m_Sema->getLangOpts().CPlusPlus) {
        // FIXME: This makes sure that we ignore the contexts associated
        // with C structs, unions, and enums when looking for a matching
        // tag declaration or definition. See the similar lookup tweak
        // in Sema::LookupName; is there a better way to deal with this?
        while (isa<RecordDecl>(SearchDC) || isa<EnumDecl>(SearchDC))
          SearchDC = SearchDC->getParent();
      }
    } else if (m_Sema->getScopeForContext(m_Sema->CurContext)
               ->isFunctionPrototypeScope()) {
      // If this is an enum declaration in function prototype scope, set its
      // initial context to the translation unit.
      SearchDC = m_Context->getTranslationUnitDecl();
    }

    if (Previous.isSingleResult() &&
        Previous.getFoundDecl()->isTemplateParameter()) {
      // Maybe we will complain about the shadowed template parameter.
      m_Sema->DiagnoseTemplateParameterShadow(NameLoc, Previous.getFoundDecl());
      // Just pretend that we didn't see the previous declaration.
      Previous.clear();
    }

    if (m_Sema->getLangOpts().CPlusPlus && Name && DC && m_Sema->StdNamespace
        && DC->Equals(m_Sema->getStdNamespace()) && Name->isStr("bad_alloc")) {
      // This is a declaration of or a reference to "std::bad_alloc".
      //isStdBadAlloc = true;

      if (Previous.empty() && m_Sema->StdBadAlloc) {
        // std::bad_alloc has been implicitly declared (but made invisible to
        // name lookup). Fill in this implicit declaration as the previous
        // declaration, so that the declarations get chained appropriately.
        Previous.addDecl(m_Sema->getStdBadAlloc());
      }
    }

    if (!Previous.empty()) {
      NamedDecl *PrevDecl = (*Previous.begin())->getUnderlyingDecl();

      // It's okay to have a tag decl in the same scope as a typedef
      // which hides a tag decl in the same scope.  Finding this
      // insanity with a redeclaration lookup can only actually happen
      // in C++.
      //
      // This is also okay for elaborated-type-specifiers, which is
      // technically forbidden by the current standard but which is
      // okay according to the likely resolution of an open issue;
      // see http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#407
      if (m_Sema->getLangOpts().CPlusPlus) {
        if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(PrevDecl)) {
          if (const TagType *TT = TD->getUnderlyingType()->getAs<TagType>()) {
            TagDecl *Tag = TT->getDecl();
            if (Tag->getDeclName() == Name &&
                Tag->getDeclContext()->getRedeclContext()
                ->Equals(TD->getDeclContext()->getRedeclContext())) {
              PrevDecl = Tag;
              Previous.clear();
              Previous.addDecl(Tag);
              Previous.resolveKind();
            }
          }
        }
      }

      if (TagDecl *PrevTagDecl = dyn_cast<TagDecl>(PrevDecl)) {
        // If this is a use of a previous tag, or if the tag is already declared
        // in the same scope (so that the definition/declaration completes or
        // rementions the tag), reuse the decl.
        if (m_Sema->isDeclInScope(PrevDecl, SearchDC,
                                 m_Sema->getScopeForContext(m_Sema->CurContext),
                                  isExplicitSpecialization)) {
          // Make sure that this wasn't declared as an enum and now used as a
          // struct or something similar.
          SourceLocation KWLoc = NewTD->getBeginLoc();
          if (!m_Sema->isAcceptableTagRedeclaration(PrevTagDecl, Kind,
                                          NewTD->isThisDeclarationADefinition(),
                                                    KWLoc, Name)) {
            bool SafeToContinue
              = (PrevTagDecl->getTagKind() != TTK_Enum && Kind != TTK_Enum);

            if (SafeToContinue)
              m_Sema->Diag(KWLoc, diag::err_use_with_wrong_tag)
                << Name
                << FixItHint::CreateReplacement(SourceRange(KWLoc),
                                                PrevTagDecl->getKindName());
            else
              m_Sema->Diag(KWLoc, diag::err_use_with_wrong_tag) << Name;
            m_Sema->Diag(PrevTagDecl->getLocation(), diag::note_previous_use);

            if (SafeToContinue)
              Kind = PrevTagDecl->getTagKind();
            else {
              // Recover by making this an anonymous redefinition.
              Name = 0;
              Previous.clear();
              Invalid = true;
            }
          }

          if (Kind == TTK_Enum && PrevTagDecl->getTagKind() == TTK_Enum) {
            const EnumDecl *NewEnum = cast<EnumDecl>(NewTD);
            const EnumDecl *PrevEnum = cast<EnumDecl>(PrevTagDecl);

            // All conflicts with previous declarations are recovered by
            // returning the previous declaration.
            if (NewEnum->isScoped() != PrevEnum->isScoped()) {
              m_Sema->Diag(KWLoc, diag::err_enum_redeclare_scoped_mismatch)
                << PrevEnum->isScoped();
              m_Sema->Diag(PrevTagDecl->getLocation(), diag::note_previous_use);

              return false;
            }
            else if (PrevEnum->isFixed()) {
              QualType T = NewEnum->getIntegerType();

              if (!m_Context->hasSameUnqualifiedType(T,
                                                  PrevEnum->getIntegerType())) {
                m_Sema->Diag(NameLoc.isValid() ? NameLoc : KWLoc,
                             diag::err_enum_redeclare_type_mismatch)
                  << T
                  << PrevEnum->getIntegerType();
                m_Sema->Diag(PrevTagDecl->getLocation(),
                             diag::note_previous_use);

                return false;
              }
            }
            else if (NewEnum->isFixed() != PrevEnum->isFixed()) {
              m_Sema->Diag(KWLoc, diag::err_enum_redeclare_fixed_mismatch)
                << PrevEnum->isFixed();
              m_Sema->Diag(PrevTagDecl->getLocation(), diag::note_previous_use);

              return false;
            }
          }

          if (!Invalid) {
            // If this is a use, just return the declaration we found.

            // Diagnose attempts to redefine a tag.
            if (NewTD->isThisDeclarationADefinition()) {
              if (TagDecl* Def = PrevTagDecl->getDefinition()) {
                // If we're defining a specialization and the previous
                // definition is from an implicit instantiation, don't emit an
                // error here; we'll catch this in the general case below.
                if (!isExplicitSpecialization ||
                    !isa<CXXRecordDecl>(Def) ||
                    cast<CXXRecordDecl>(Def)->getTemplateSpecializationKind()
                    == TSK_ExplicitSpecialization) {
                  m_Sema->Diag(NameLoc, diag::err_redefinition) << Name;
                  m_Sema->Diag(Def->getLocation(),
                               diag::note_previous_definition);
                  // If this is a redefinition, recover by making this
                  // struct be anonymous, which will make any later
                  // references get the previous definition.
                  Name = 0;
                  Previous.clear();
                  Invalid = true;
                }
              } else {
                // If the type is currently being defined, complain
                // about a nested redefinition.
                const TagType *Tag
                  = cast<TagType>(m_Context->getTagDeclType(PrevTagDecl));
                if (Tag->isBeingDefined()) {
                  m_Sema->Diag(NameLoc, diag::err_nested_redefinition) << Name;
                  m_Sema->Diag(PrevTagDecl->getLocation(),
                               diag::note_previous_definition);
                  Name = 0;
                  Previous.clear();
                  Invalid = true;
                }
              }

              // Okay, this is definition of a previously declared or referenced
              // tag PrevDecl. We're going to create a new Decl for it.
            }
          }
          // If we get here we have (another) forward declaration or we
          // have a definition.  Just create a new decl.

        } else {
          // If we get here, this is a definition of a new tag type in a nested
          // scope, e.g. "struct foo; void bar() { struct foo; }", just create a
          // new decl/type.  We set PrevDecl to NULL so that the entities
          // have distinct types.
          Previous.clear();
        }
        // If we get here, we're going to create a new Decl. If PrevDecl
        // is non-NULL, it's a definition of the tag declared by
        // PrevDecl. If it's NULL, we have a new definition.


        // Otherwise, PrevDecl is not a tag, but was found with tag
        // lookup.  This is only actually possible in C++, where a few
        // things like templates still live in the tag namespace.
      } else {
        assert(m_Sema->getLangOpts().CPlusPlus);

        // Diagnose if the declaration is in scope.
        if (!m_Sema->isDeclInScope(PrevDecl, SearchDC,
                                 m_Sema->getScopeForContext(m_Sema->CurContext),
                                   isExplicitSpecialization)) {
          // do nothing

          // Otherwise it's a declaration.  Call out a particularly common
          // case here.
        } else if (TypedefNameDecl *TND = dyn_cast<TypedefNameDecl>(PrevDecl)) {
          unsigned Kind = 0;
          if (isa<TypeAliasDecl>(PrevDecl)) Kind = 1;
          m_Sema->Diag(NameLoc, diag::err_tag_definition_of_typedef)
            << Name << Kind << TND->getUnderlyingType();
          m_Sema->Diag(PrevDecl->getLocation(),
                       diag::note_previous_decl) << PrevDecl;
          Invalid = true;

          // Otherwise, diagnose.
        } else {
          // The tag name clashes with something else in the target scope,
          // issue an error and recover by making this tag be anonymous.
          m_Sema->Diag(NameLoc, diag::err_redefinition_different_kind) << Name;
          m_Sema->Diag(PrevDecl->getLocation(), diag::note_previous_definition);
          Name = 0;
          Invalid = true;
        }

        // The existing declaration isn't relevant to us; we're in a
        // new scope, so clear out the previous declaration.
        Previous.clear();
      }
    }
    if (Invalid) {
      return false;
    }

    return true;
  }
} // namespace cling
