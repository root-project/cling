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

  DefinitionShadower::DefinitionShadower(Sema& S, Interpreter& I)
          : ASTTransformer(&S), m_Interp(I), m_Context(S.getASTContext()),
            m_TU(S.getASTContext().getTranslationUnitDecl())
  {}

  bool DefinitionShadower::isClingShadowNamespace(const DeclContext *DC) {
    auto NS = dyn_cast<NamespaceDecl>(DC);
    return NS && NS->getName().startswith("__cling_N5");
  }

  void DefinitionShadower::hideDecl(clang::NamedDecl *D) const {
    assert(isClingShadowNamespace(D->getDeclContext())
             && "D not in a __cling_N5xxx namespace?");

    // FIXME: this hides a decl from SemaLookup (there is no unloading). For
    // (large) L-values, this might be a memory leak. Should this be fixed?
    if (Scope* S = m_Sema->getScopeForContext(m_TU)) {
      S->RemoveDecl(D);
      if (utils::Analyze::isOnScopeChains(D, *m_Sema))
        m_Sema->IdResolver.RemoveDecl(D);
    }
    clang::StoredDeclsList &SDL = (*m_TU->getLookupPtr())[D->getDeclName()];
    if (SDL.getAsVector() || SDL.getAsDecl() == D)
      SDL.remove(D);

    if (InterpreterCallbacks *IC = m_Interp.getCallbacks())
      IC->DefinitionShadowed(D);
  }

  void DefinitionShadower::invalidatePreviousDefinitions(NamedDecl *D) const {
    LookupResult Previous(*m_Sema, D->getDeclName(), D->getLocation(),
                   Sema::LookupOrdinaryName, Sema::ForRedeclaration);
    m_Sema->LookupQualifiedName(Previous, m_TU);

    for (auto Prev : Previous) {
      if (Prev == D)
        continue;
      if (isDefinition(Prev)
          && (!isDefinition(D) || !isClingShadowNamespace(Prev->getDeclContext())))
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
    const CompilationOptions &CO = T->getCompilationOpts();
    // Global declarations whose origin is the Cling prompt are subject to be
    // nested in a `__cling_N5' namespace.
    if (!CO.EnableShadowing
        || D->getLexicalDeclContext() != m_TU || D->isInvalidDecl()
        || isa<UsingDirectiveDecl>(D) || isa<UsingDecl>(D)
        // FIXME: NamespaceDecl requires additional processing (TBD)
        || isa<NamespaceDecl>(D)
        || (isa<FunctionDecl>(D) && cast<FunctionDecl>(D)->isTemplateInstantiation())
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
