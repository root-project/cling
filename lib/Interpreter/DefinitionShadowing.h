//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Javier López-Gómez <javier.lopez.gomez@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_DEFINITION_SHADOWING
#define CLING_DEFINITION_SHADOWING

#include "ASTTransformer.h"

namespace clang {
  class ASTContext;
  class Decl;
  class TranslationUnitDecl;
  class NamedDecl;
  class FunctionDecl;
  class Sema;
}

namespace cling {
  class DefinitionShadowing : public ASTTransformer {
  private:
    clang::ASTContext          *m_Context;
    clang::TranslationUnitDecl *m_TU;

    void hideDecl(clang::NamedDecl *D);
    bool invalidatePreviousDefinitions(clang::NamedDecl *D);
    void invalidatePreviousDefinitions(clang::FunctionDecl *D);
  public:
    DefinitionShadowing(clang::Sema* S);
    Result Transform(clang::Decl* D) override;
  };
} // end namespace cling

#endif // CLING_DEFINITION_SHADOWING
