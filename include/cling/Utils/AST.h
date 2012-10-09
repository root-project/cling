//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_UTILS_AST_H
#define CLING_UTILS_AST_H

#include "llvm/ADT/SmallSet.h"

namespace clang {
  class ASTContext;
  class Expr;
  class DeclContext;
  class DeclarationName;
  class NamedDecl;
  class NamespaceDecl;
  class QualType;
  class Sema;
  class Type;
}

namespace cling {
namespace utils {
  ///\brief Class containing static utility functions synthesizing AST nodes or
  /// types.
  ///
  class Synthesize {
  public:

    ///\brief Synthesizes c-style cast in the AST from given pointer and type to
    /// cast to.
    ///
    static clang::Expr* CStyleCastPtrExpr(clang::Sema* S,
                                          clang::QualType Ty, uint64_t Ptr);
  };

  ///\brief Class containing static utility functions transforming AST nodes or
  /// types.
  ///
  class Transform {
  public:

    ///\brief "Desugars" a type while skipping the ones in the set.
    ///
    /// Desugars a given type recursively until strips all sugar or until gets a
    /// sugared type, which is to be skipped.
    ///\param[in] Ctx - The ASTContext.
    ///\param[in] QT - The type to be partially desugared.
    ///\param[in] TypesToSkip - The set of sugared types which shouldn't be 
    ///                         desugared.
    ///\param[in] fullyQualify - if true insert Elaborated where needed.
    ///\returns Partially desugared QualType
    ///
    static clang::QualType GetPartiallyDesugaredType(const clang::ASTContext& Ctx, 
                                                     clang::QualType QT,
                        const llvm::SmallSet<const clang::Type*, 4>& TypesToSkip,
                                                     bool fullyQualify = true);

  };

  ///\brief Class containing static utility functions looking up names. Very
  /// useful for quick, simple lookups.
  /// 
  class Lookup {
  public:

    ///\brief Quick lookup for a single namespace declaration in a given 
    /// declaration context.
    ///
    ///\param[in] S - Semantic Analysis object doing the lookup.
    ///\param[in] Name - The name we are looking up.
    ///\param[in] Within - The context within the lookup is done. If 0 the 
    ///                    TranslationUnitDecl is used.
    ///\returns the found result (if single) or 0.
    ///
    static clang::NamespaceDecl* Namespace(clang::Sema* S,
                                           const char* Name,
                                           clang::DeclContext* Within = 0);

    ///\brief Quick lookup for a single named declaration in a given 
    /// declaration context.
    ///
    ///\param[in] S - Semantic Analysis object doing the lookup.
    ///\param[in] Name - The name we are looking up.
    ///\param[in] Within - The context within the lookup is done. If 0 the 
    ///                    TranslationUnitDecl is used.
    ///\returns the found result (if single) or 0.
    ///
    static clang::NamedDecl* Named(clang::Sema* S,
                                   const char* Name,
                                   clang::DeclContext* Within = 0);

    ///\brief Quick lookup for a single namespace declaration in a given 
    /// declaration context.
    ///
    ///\param[in] S - Semantic Analysis object doing the lookup.
    ///\param[in] Name - The name we are looking up. The & avoids inclusion of 
    ///                  DeclarationName.h (faster at runtime).
    ///\param[in] Within - The context within the lookup is done. If 0 the 
    ///                    TranslationUnitDecl is used.
    ///\returns the found result (if single) or 0.
    ///
    static clang::NamedDecl* Named(clang::Sema* S,
                                   const clang::DeclarationName& Name,
                                   clang::DeclContext* Within = 0);

  };
} // end namespace utils
} // end namespace cling
#endif // CLING_UTILS_AST_H
