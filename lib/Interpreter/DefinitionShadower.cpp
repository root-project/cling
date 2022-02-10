//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Javier López-Gómez <jalopezg@inf.uc3m.es>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "DefinitionShadower.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Utils/AST.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/Stmt.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include <algorithm>

using namespace clang;

namespace cling {
  /// \brief Returns whether the given source location is a Cling input line. If
  /// it came from the prompt, the file is a virtual file with overriden contents.
  static bool typedInClingPrompt(FullSourceLoc L) {
    if (L.isInvalid())
      return false;
    const SourceManager &SM = L.getManager();
    const FileID FID = SM.getFileID(L);
    return SM.isFileOverridden(SM.getFileEntryForID(FID))
           && (SM.getFileID(SM.getIncludeLoc(FID)) == SM.getMainFileID());
  }

  /// \brief Returns whether the given {Function,Tag,Var}Decl/TemplateDecl is a definition.
  static bool isDefinition(const Decl *D) {
    if (auto FD = dyn_cast<FunctionDecl>(D))
      return FD->isThisDeclarationADefinition();
    if (auto TD = dyn_cast<TagDecl>(D))
      return TD->isThisDeclarationADefinition();
    if (auto VD = dyn_cast<VarDecl>(D))
      return VD->isThisDeclarationADefinition();
    if (auto TD = dyn_cast<TemplateDecl>(D))
      return isDefinition(TD->getTemplatedDecl());
    return true;
  }

  /// \brief Returns whether the given declaration is a template instantiation
  /// or specialization.
  static bool isInstantiationOrSpecialization(const Decl *D) {
    if (auto FD = dyn_cast<FunctionDecl>(D))
      return FD->isTemplateInstantiation() || FD->isFunctionTemplateSpecialization();
    if (auto CTSD = dyn_cast<ClassTemplateSpecializationDecl>(D))
      return CTSD->getSpecializationKind() != TSK_Undeclared;
    if (auto VTSD = dyn_cast<VarTemplateSpecializationDecl>(D))
      return VTSD->getSpecializationKind() != TSK_Undeclared;
    return false;
  }

  DefinitionShadower::DefinitionShadower(Sema& S, Interpreter& I)
          : ASTTransformer(&S), m_Context(S.getASTContext()), m_Interp(I),
            m_TU(S.getASTContext().getTranslationUnitDecl()),
            m_ShadowsDeclInStdDiagID(S.getDiagnostics().getCustomDiagID(
                DiagnosticsEngine::Warning,
                "'%0' shadows a declaration with the same name in the 'std' "
                "namespace; use '::%0' to reference this declaration"))
  {}

  bool DefinitionShadower::isClingShadowNamespace(const DeclContext *DC) {
    auto NS = dyn_cast<NamespaceDecl>(DC);
    return NS && NS->getName().startswith("__cling_N5");
  }

  void DefinitionShadower::hideDecl(clang::NamedDecl *D) const {
    // FIXME: this hides a decl from SemaLookup (there is no unloading). For
    // (large) L-values, this might be a memory leak. Should this be fixed?
    if (Scope* S = m_Sema->getScopeForContext(m_TU)) {
      S->RemoveDecl(D);
      if (utils::Analyze::isOnScopeChains(D, *m_Sema))
        m_Sema->IdResolver.RemoveDecl(D);
    }
    clang::StoredDeclsList &SDL = (*m_TU->getLookupPtr())[D->getDeclName()];
    if (SDL.getAsDecl() == D) {
      SDL.setOnlyValue(nullptr);
    }
    if (auto Vec = SDL.getAsVector()) {
      // FIXME: investigate why StoredDeclList has duplicated entries coming from PCM.
      Vec->erase(std::remove_if(Vec->begin(), Vec->end(),
                                [D](Decl *Other) { return cast<Decl>(D) == Other; }),
                 Vec->end());
    }

    if (InterpreterCallbacks *IC = m_Interp.getCallbacks())
      IC->DefinitionShadowed(D);
  }

  void DefinitionShadower::invalidatePreviousDefinitions(NamedDecl *D) const {
    // NotForRedeclaration: lookup anything visible; follows using directives
    LookupResult Previous(*m_Sema, D->getDeclName(), D->getLocation(),
                   Sema::LookupOrdinaryName, Sema::NotForRedeclaration);
    Previous.suppressDiagnostics();
    m_Sema->LookupQualifiedName(Previous, m_TU);
    bool shadowsDeclInStd = false;

    for (auto Prev : Previous) {
      if (Prev == D)
        continue;
      if (isDefinition(Prev) && !isDefinition(D))
        continue;
      // If the found declaration is a function overload, do not invalidate it.
      // For templated functions, Sema::IsOverload() does the right thing as per
      // C++ [temp.over.link]p4.
      if (isa<FunctionDecl>(Prev) && isa<FunctionDecl>(D)
          && m_Sema->IsOverload(cast<FunctionDecl>(D),
                                cast<FunctionDecl>(Prev), /*IsForUsingDecl=*/false))
        continue;
      if (isa<FunctionTemplateDecl>(Prev) && isa<FunctionTemplateDecl>(D)
          && m_Sema->IsOverload(cast<FunctionTemplateDecl>(D)->getTemplatedDecl(),
                                cast<FunctionTemplateDecl>(Prev)->getTemplatedDecl(),
                                /*IsForUsingDecl=*/false))
        continue;

      shadowsDeclInStd |= Prev->isInStdNamespace();
      hideDecl(Prev);

      // For unscoped enumerations, also invalidate all enumerators.
      if (EnumDecl *ED = dyn_cast<EnumDecl>(Prev)) {
        if (!ED->isTransparentContext())
          continue;
        for (auto &J : ED->decls())
          if (NamedDecl *ND = dyn_cast<NamedDecl>(J))
            hideDecl(ND);
      }
    }

    // Ignore any forward declaration issued after a definition. Fixes "error
    // : reference to 'xxx' is ambiguous" in `class C {}; class C; C foo;`.
    if (!Previous.empty() && !isDefinition(D))
      D->setInvalidDecl();

    // Diagnose shadowing of decls in the `std` namespace (see ROOT-5971).
    // For unnamed macros, the input is ingested in a single `Interpreter::process()`
    // call. Do not emit the warning in that case, as all references are local
    // to the wrapper function and this diagnostic might be misleading.
    if (shadowsDeclInStd
        && ((m_Interp.getInputFlags() & (Interpreter::kInputFromFile
                                         | Interpreter::kIFFLineByLine))
             != Interpreter::kInputFromFile)) {
      m_Sema->Diag(D->getBeginLoc(), m_ShadowsDeclInStdDiagID)
        << D->getQualifiedNameAsString();
    }
  }

  void DefinitionShadower::invalidatePreviousDefinitions(FunctionDecl *D) const {
    const CompilationOptions &CO = getTransaction()->getCompilationOpts();
    if (utils::Analyze::IsWrapper(D)) {
      if (!CO.DeclarationExtraction)
        return;

      // DeclExtractor shall move local declarations to the TU. Invalidate all
      // previous definitions (that may clash) before it runs.
      auto CS = dyn_cast<CompoundStmt>(D->getBody());
      for (auto &I : CS->body()) {
        auto DS = dyn_cast<DeclStmt>(I);
        if (!DS)
          continue;
        for (auto &J : DS->decls())
          if (auto ND = dyn_cast<NamedDecl>(J))
            invalidatePreviousDefinitions(ND);
      }
    } else
      invalidatePreviousDefinitions(cast<NamedDecl>(D));
  }

  void DefinitionShadower::invalidatePreviousDefinitions(Decl *D) const {
    if (auto FD = dyn_cast<FunctionDecl>(D))
      invalidatePreviousDefinitions(FD);
    else if (auto ND = dyn_cast<NamedDecl>(D))
      invalidatePreviousDefinitions(ND);
  }

  ASTTransformer::Result DefinitionShadower::Transform(Decl* D) {
    Transaction *T = getTransaction();
    if (!T->getCompilationOpts().EnableShadowing)
      return Result(D, true);

    // For variable templates, Transform() is invoked with a VarDecl; get the
    // corresponding VarTemplateDecl.
    if (auto VD = dyn_cast<VarDecl>(D))
      if (auto VTD = VD->getDescribedVarTemplate())
        D = VTD;

    // Disable definition shadowing for some specific cases.
    if (D->getLexicalDeclContext() != m_TU || D->isInvalidDecl()
        || isa<UsingDirectiveDecl>(D) || isa<UsingDecl>(D) || isa<NamespaceDecl>(D)
        || isInstantiationOrSpecialization(D)
        || !typedInClingPrompt(FullSourceLoc{D->getLocation(),
                                             m_Context.getSourceManager()}))
      return Result(D, true);

    // Each transaction gets at most a `__cling_N5xxx' namespace. If `T' already
    // has one, reuse it.
    auto NS = T->getDefinitionShadowNS();
    if (!NS) {
      NS = NamespaceDecl::Create(m_Context, m_TU, /*inline=*/true,
                                 SourceLocation(), SourceLocation(),
                                 &m_Context.Idents.get("__cling_N5"
                                       + std::to_string(m_UniqueNameCounter++)),
                                 nullptr);
      //NS->setImplicit();
      m_TU->addDecl(NS);
      T->setDefinitionShadowNS(NS);
    }

    m_TU->removeDecl(D);
    if (isa<CXXRecordDecl>(D->getDeclContext()))
      D->setLexicalDeclContext(NS);
    else
      D->setDeclContext(NS);
    // An instantiated function template inherits the declaration context of the
    // templated decl. This is used for name mangling; fix it to avoid clashing.
    if (auto FTD = dyn_cast<FunctionTemplateDecl>(D))
      FTD->getTemplatedDecl()->setDeclContext(NS);
    NS->addDecl(D);

    // Invalidate previous definitions so that LookupResult::resolveKind() does not
    // mark resolution as ambiguous.
    invalidatePreviousDefinitions(D);
    return Result(D, true);
  }
} // end namespace cling
