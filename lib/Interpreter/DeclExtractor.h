//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_DECL_EXTRACTOR_H
#define CLING_DECL_EXTRACTOR_H

#include "TransactionTransformer.h"

#include "clang/Sema/Lookup.h"

namespace clang {
  class ASTContext;
  class Decl;
  class DeclContext;
  class NamedDecl;
  class Scope;
}

namespace cling {
  class DeclExtractor : public TransactionTransformer {
  private:
    clang::ASTContext* m_Context;
  public:
    DeclExtractor(clang::Sema* S);

    virtual ~DeclExtractor();

    ///\brief Iterates over the transaction and finds cling specific wrappers.
    /// Scans the wrappers for declarations and extracts them onto the global
    /// scope.
    ///
    virtual void Transform();

  private:

    ///\brief Tries to extract the declaration on the global scope (translation
    /// unit scope).
    ///
    ///\param D[in] - The function declaration to extract from.
    ///\returns true on success.
    ///
    bool ExtractDecl(clang::FunctionDecl* FD);

    ///\brief Checks for clashing names when trying to extract a declaration.
    ///
    ///\returns true if there is another declaration with the same name
    ///
    bool CheckForClashingNames(
                           const llvm::SmallVector<clang::NamedDecl*, 4>& Decls,
                               clang::DeclContext* DC, clang::Scope* S);

    ///\brief Performs semantic checking on a newly-extracted tag declaration.
    ///
    /// This routine performs all of the type-checking required for a tag
    /// declaration once it has been built. It is used both to check tags before
    /// they have been moved onto the global scope.
    ///
    /// Sets NewTD->isInvalidDecl if an error was encountered.
    ///
    ///\returns true if the tag declaration is redeclaration.
    ///
    bool CheckTagDeclaration(clang::TagDecl* NewTD,
                             clang::LookupResult& Previous);
  };

} // namespace cling

#endif // CLING_DECL_EXTRACTOR_H
