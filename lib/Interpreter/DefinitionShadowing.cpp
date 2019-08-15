//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Javier López-Gómez <javier.lopez.gomez@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "DefinitionShadowing.h"
#include <iostream>

#include "cling/Utils/AST.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/Stmt.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace cling {
  /// \brief Returns whether `L' comes from a Cling input line.
  static bool typedInClingPrompt(FullSourceLoc L) {
    if (L.isInvalid())
      return false;
    llvm::StringRef S{L.getManager().getPresumedLoc(L).getFilename()};
    return S.startswith("ROOT_prompt_")
           || S.startswith("input_line_");
  }

  /// \brief Returns whether the given {Function,Tag,Var}Decl is a definition.
  static bool isDefinition(Decl *D) {
    if (auto _D = dyn_cast<FunctionDecl>(D)) return _D->isThisDeclarationADefinition();
    if (auto _D = dyn_cast<TagDecl>(D)) return _D->isThisDeclarationADefinition();
    if (auto _D = dyn_cast<VarDecl>(D)) return _D->isThisDeclarationADefinition();
    return true;
  }

  DefinitionShadowing::DefinitionShadowing(Sema* S)
    : ASTTransformer(S), m_Context(&S->getASTContext()),
      m_TU(S->getASTContext().getTranslationUnitDecl())
  {
    const_cast<PrintingPolicy&>(m_Context->getPrintingPolicy()).SuppressUnwrittenScope = true;
  }

  /// \brief Make a declaration hidden for SemaLookup; internally used in `invalidatePreviousDefinitions()'.
  /// This directly manipulates lookup tables to avoid a patch to Clang!
  void DefinitionShadowing::hideDecl(clang::NamedDecl *D) {
    if (Scope* S = m_Sema->getScopeForContext(m_TU)) {
      S->RemoveDecl(D);
      if (utils::Analyze::isOnScopeChains(D, *m_Sema))
        m_Sema->IdResolver.RemoveDecl(D);
    }
    clang::StoredDeclsList &SDL = (*m_TU->getLookupPtr())[D->getDeclName()];
    if (SDL.getAsVector() || SDL.getAsDecl() == D)
      SDL.remove(D);
  }

  /// \brief Lookup the given name and invalidate all clashing declarations (as seen from the TU).
  /// \return Returns whether a previous definition exists.
  bool DefinitionShadowing::invalidatePreviousDefinitions(NamedDecl *D) {
    LookupResult Previous(*m_Sema, D->getDeclName(), D->getLocation(),
                   Sema::LookupOrdinaryName, Sema::ForRedeclaration);
    m_Sema->LookupQualifiedName(Previous, m_TU);

    for (auto I : Previous) {
      if (I == D) continue;
      auto NS = dyn_cast<NamespaceDecl>(I->getLexicalDeclContext());
      if (isDefinition(I)
          && (!isDefinition(D) || !(NS && NS->getName().startswith("__cling_N5")))) continue;

      hideDecl(I);

      if (TagDecl *TD = dyn_cast<TagDecl>(I)) {
        if (!TD->isTransparentContext()) continue;
        for (auto J : TD->decls()) hideDecl(cast<NamedDecl>(J));
      }
    }
    return !Previous.empty();
  }

  /// \brief Invalidate previous function definition.  Local declararations might be
  /// moved by DeclExtractor; in that case, invalidate all those before DeclExtractor runs.
  void DefinitionShadowing::invalidatePreviousDefinitions(FunctionDecl *D) {
    const CompilationOptions &CO = getTransaction()->getCompilationOpts();
    if (CO.DeclarationExtraction) {
      // XXX: this should make DeclExtractor behave as usual
      auto CS = dyn_cast<CompoundStmt>(D->getBody());
      for (auto I = CS->body_begin(), EI = CS->body_end(); I != EI; ++I) {
        auto DS = dyn_cast<DeclStmt>(*I);
        if (!DS) continue;

        for (auto J = DS->decl_begin(); J != DS->decl_end(); ++J)
          if (auto ND = dyn_cast<NamedDecl>(*J))
            // Ignore any forward declaration issued after a definition. Fixes "error
            // : reference to 'xxx' is ambiguous" in `class C {}; class C; C foo;`. 
            if (invalidatePreviousDefinitions(ND) && !isDefinition(ND))
              ND->setInvalidDecl();
      }
    } else
      invalidatePreviousDefinitions(cast<NamedDecl>(D));
  }

  ASTTransformer::Result DefinitionShadowing::Transform(Decl* D) {
    auto FD = dyn_cast<FunctionDecl>(D);
    const CompilationOptions &CO = getTransaction()->getCompilationOpts();
    // Declarations whose origin is the Cling prompt are subject to be nested in a
    // `__cling_N5' namespace.
    if (D->getLexicalDeclContext() != m_TU
        // FIXME: NamespaceDecl/TemplateDecl require additional processing (TBD)
        || isa<NamespaceDecl>(D) || D->isTemplateDecl() || (FD && FD->isFunctionTemplateSpecialization())
        || isa<UsingDirectiveDecl>(D)
        || !typedInClingPrompt(FullSourceLoc{D->getLocation(),
                                             m_Context->getSourceManager()})
        || !CO.IgnorePromptDiags /* XXX: raw input */)
     return Result(D, true);

    NamespaceDecl *NS = NamespaceDecl::Create(*m_Context, m_TU, /*inline=*/true, 
                                              SourceLocation(), SourceLocation(),
                                              &m_Context->Idents.get("__cling_N5"
                                                + std::to_string((unsigned long)D)),
                                              nullptr);
    m_TU->removeDecl(D);
    if (isa<CXXRecordDecl>(D->getDeclContext()))
      D->setLexicalDeclContext(NS);
    else
      D->setDeclContext(NS);
    NS->addDecl(D);
    m_TU->addDecl(NS);
    //NS->setImplicit();

    // Invalidate previous definitions so that LookupResult::resolveKind() does not
    // mark resolution as ambiguous.
    if (auto _D = dyn_cast<TagDecl>(D))
      invalidatePreviousDefinitions(_D);
    else if (auto _D = dyn_cast<FunctionDecl>(D))
      invalidatePreviousDefinitions(_D);

    // Ensure `NS` is unloaded from the AST on transaction rollback, e.g. '.undo X'
    getTransaction()->append(NS);
    return Result(D, true);
  }
} // end namespace cling
