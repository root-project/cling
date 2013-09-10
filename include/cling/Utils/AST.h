//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_UTILS_AST_H
#define CLING_UTILS_AST_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
  class ASTContext;
  class Expr;
  class DeclContext;
  class DeclarationName;
  class FunctionDecl;
  class IntegerLiteral;
  class NamedDecl;
  class NamespaceDecl;
  class QualType;
  class Sema;
  class Type;
}

namespace cling {
namespace utils {

  ///\brief Class containing static utility functions analizing ASTNodes or 
  /// types.
  class Analyze {
  public:

    ///\brief Checks whether the declaration is a interpreter-generated wrapper
    /// function.
    ///
    ///\param[in] ND - The decl being checked. If null returns false. 
    ///
    ///\returns true if the decl is a interpreter-generated wrapper function.
    ///
    static bool IsWrapper(const clang::NamedDecl* ND);

    ///\brief Retrieves the last expression of a function body. If it was a
    /// DeclStmt with a variable declaration, creates DeclRefExpr and adds it to
    /// the function body.
    ///
    /// Useful for value printing (deciding where to attach the value printer)
    /// and value evaluation (deciding that is the type of a value)
    ///
    ///\param[in] FD            - The declaration being analyzed. 
    ///\param[in] FoundAt       - The position of the expression to be returned
    ///                           in function's body.
    ///\param[in] omitDeclStmts - Whether or not to synthesize DeclRefExpr if
    ///                           there is DeclStmt.
    ///\param[in] S             - The semantic analysis object used for 
    ///                           synthesis of the DeclRefExpr. 
    ///\returns 0 if the operation wasn't successful.
    ///
    static clang::Expr* GetOrCreateLastExpr(clang::FunctionDecl* FD, 
                                            int* FoundAt = 0,
                                            bool omitDeclStmts = true,
                                            clang::Sema* S = 0);
  };

  ///\brief Class containing static utility functions synthesizing AST nodes or
  /// types.
  ///
  class Synthesize {
  public:
    static const char* const UniquePrefix;

    ///\brief Synthesizes c-style cast in the AST from given pointer and type to
    /// cast to.
    ///
    static clang::Expr* CStyleCastPtrExpr(clang::Sema* S, clang::QualType Ty, 
                                          uint64_t Ptr);

    ///\brief Synthesizes c-style cast in the AST from given pointer and type to
    /// cast to.
    ///
    static clang::Expr* CStyleCastPtrExpr(clang::Sema* S, clang::QualType Ty, 
                                          clang::Expr* E);

    ///\brief Synthesizes integer literal given an unsigned.
    ///
    //  TODO: Use Sema::ActOnIntegerConstant.
    static clang::IntegerLiteral* IntegerLiteralExpr(clang::ASTContext& C, 
                                                     uint64_t Ptr);

  };

  ///\brief Class containing static utility functions transforming AST nodes or
  /// types.
  ///
  class Transform {
  public:

    ///\brief Class containing the information on how to configure the
    /// transformation
    ///
    struct Config {
      typedef const clang::Type cType;
      typedef llvm::SmallSet<cType*, 4> SkipCollection;
      typedef llvm::DenseMap<cType*, cType*> ReplaceCollection;
      
      SkipCollection    m_toSkip;
      ReplaceCollection m_toReplace;

      bool empty() const { return m_toSkip.size()==0 && m_toReplace.empty(); }
    };
     
    ///\brief Remove one layer of sugar, but only some kinds.
    static bool SingleStepPartiallyDesugarType(clang::QualType& QT,
                                               const clang::ASTContext& C);

    ///\brief "Desugars" a type while skipping the ones in the set.
    ///
    /// Desugars a given type recursively until strips all sugar or until gets a
    /// sugared type, which is to be skipped.
    ///\param[in] Ctx - The ASTContext.
    ///\param[in] QT - The type to be partially desugared.
    ///\param[in] TypeConfig - The set of sugared types which shouldn't be 
    ///                        desugared and those that should be replaced.
    ///\param[in] fullyQualify - if true insert Elaborated where needed.
    ///\returns Partially desugared QualType
    ///
    static
    clang::QualType
    GetPartiallyDesugaredType(const clang::ASTContext& Ctx, clang::QualType QT,
      const Config& TypeConfig,
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
                                          const clang::DeclContext* Within = 0);

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
                                   const clang::DeclContext* Within = 0);

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
                                   const clang::DeclContext* Within = 0);

  };
} // end namespace utils
} // end namespace cling
#endif // CLING_UTILS_AST_H
