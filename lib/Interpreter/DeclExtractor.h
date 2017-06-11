//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_DECL_EXTRACTOR_H
#define CLING_DECL_EXTRACTOR_H

#include "ASTTransformer.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace clang {
  class ASTContext;
  class DeclContext;
  class DeclGroupRef;
  class FunctionDecl;
  class LookupResult;
  class NamedDecl;
  class Scope;
  class Sema;
  class Stmt;
  class TagDecl;
}

namespace cling {
  class DeclExtractor : public WrapperTransformer {
  private:
    clang::ASTContext* m_Context;

    /// \brief Counter used when we need unique names.
    unsigned long long m_UniqueNameCounter;

  public:
    DeclExtractor(clang::Sema* S);

    virtual ~DeclExtractor();

    ///\brief Scans the wrapper for declarations and extracts them onto the
    /// global scope.
    ///
    Result Transform(clang::Decl* D) override;

  private:

    ///\brief Tries to extract the declaration on the global scope (translation
    /// unit scope).
    ///
    ///\param D[in] - The function declaration to extract from.
    ///\returns true on success.
    ///
    bool ExtractDecl(clang::FunctionDecl* FD);

    /// \brief Creates unique name (eg. of a variable). Used internally for
    /// AST node synthesis.
    ///
    void createUniqueName(std::string& out);

    ///\brief Enforces semantically correct initialization order.
    ///
    /// If we consider \code int i = 1; i++; int j = i; \endcode the code
    /// snippet will be transformed into
    /// \code int i; int j = i; void __cling_wrapper() { int++ } \endcode and
    /// the result of will be 1 and not 2. This function scans whether there is
    /// more than one declaration and generates:
    /// \code
    /// int i = 1;
    /// int __fd_init_order__cling_Un1Qu30() {
    ///   i++;
    /// }
    /// int __vd_init_order__cling_Un1Qu31 = __fd_init_order__cling_Un1Qu30();
    /// int j = i;
    /// \endcode
    ///
    ///\param[in] Stmts - Collection for which have to run as part of the
    ///                   static initialization.
    ///
    void EnforceInitOrder(llvm::SmallVectorImpl<clang::Stmt*>& Stmts);

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


    ///\brief Validate a variable that is a CXXRecordDecl
    ///
    /// Currently only reports errors if the var is a lamda that captures by
    /// copy.
    ///
    ///\returns whether an error was reported
    ///
    bool ValidateCXXRecord(clang::VarDecl* VD) const;
  };

} // namespace cling

#endif // CLING_DECL_EXTRACTOR_H
