//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_UTILS_AST_H
#define CLING_UTILS_AST_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
  class ASTContext;
  class Expr;
  class Decl;
  class DeclContext;
  class DeclarationName;
  class GlobalDecl;
  class FunctionDecl;
  class IntegerLiteral;
  class NamedDecl;
  class NamespaceDecl;
  class NestedNameSpecifier;
  class QualType;
  class Sema;
  class TagDecl;
  class TemplateDecl;
  class Type;
  class TypedefNameDecl;
}

namespace cling {
namespace utils {

  ///\brief Class containing static utility functions analizing ASTNodes or
  /// types.
  namespace Analyze {

    ///\brief Checks whether the declaration is a interpreter-generated wrapper
    /// function.
    ///
    ///\param[in] ND - The decl being checked. If null returns false.
    ///
    ///\returns true if the decl is a interpreter-generated wrapper function.
    ///
    bool IsWrapper(const clang::NamedDecl* ND);

    ///\brief Get the mangled name of a GlobalDecl.
    ///
    ///\param [in]  GD - try to mangle this decl's name.
    ///\param [out] mangledName - put the mangled name in here.
    ///
    void maybeMangleDeclName(const clang::GlobalDecl& GD,
                             std::string& mangledName);


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
    clang::Expr* GetOrCreateLastExpr(clang::FunctionDecl* FD,
                                     int* FoundAt = 0,
                                     bool omitDeclStmts = true,
                                     clang::Sema* S = 0);

    ///\brief Return true if the class or template is declared directly in the
    /// std namespace (modulo inline namespace).
    ///
    ///\param[in] decl          - The declaration being analyzed.
    bool IsStdClass(const clang::NamedDecl &decl);

    ///\brief Return true if the decl has been declared in ths std namespace
    /// or is a compiler details (in __gnu_cxx and starting with a leading
    /// underscore).
    ///
    ///\param[in] decl          - The declaration being analyzed.
    bool IsStdOrCompilerDetails(const clang::NamedDecl &decl);

    ///\brief Checks whether the declaration was pushed onto the declaration
    /// chains.
    ///\param[in] ND - The declaration that is being checked.
    ///\param[in] SemaR - Sema.
    ///
    ///\returns true if the ND was found in the lookup chain.
    ///
    // See Sema::PushOnScopeChains
    ///
    bool isOnScopeChains(const clang::NamedDecl* ND, clang::Sema& SemaR);
  }

  ///\brief Class containing static utility functions synthesizing AST nodes or
  /// types.
  ///
  namespace Synthesize {

    extern const char* const UniquePrefix;

    ///\brief Synthesizes c-style cast in the AST from given pointer and type to
    /// cast to.
    ///
    clang::Expr* CStyleCastPtrExpr(clang::Sema* S, clang::QualType Ty,
                                   uint64_t Ptr);

    ///\brief Synthesizes c-style cast in the AST from given pointer and type to
    /// cast to.
    ///
    clang::Expr* CStyleCastPtrExpr(clang::Sema* S, clang::QualType Ty,
                                   clang::Expr* E);

    ///\brief Synthesizes integer literal given an unsigned.
    ///
    //  TODO: Use Sema::ActOnIntegerConstant.
    clang::IntegerLiteral* IntegerLiteralExpr(clang::ASTContext& C,
                                              uint64_t Ptr);

  }

  ///\brief Class containing static utility functions transforming AST nodes or
  /// types.
  ///
  namespace Transform {

    ///\brief Class containing the information on how to configure the
    /// transformation
    ///
    struct Config {
      typedef llvm::SmallSet<const clang::Decl*, 4> SkipCollection;
      typedef const clang::Type cType;
      typedef llvm::DenseMap<cType*, cType*> ReplaceCollection;

      SkipCollection    m_toSkip;
      ReplaceCollection m_toReplace;

      ///\brief Returns the number of default argument that should be dropped.
      /// from the name of the template instances.
      ///
      ///\param[in] templateDecl   - The declaration being analyzed.
      unsigned int DropDefaultArg(clang::TemplateDecl &templateDecl) const;

      bool empty() const { return m_toSkip.size()==0 && m_toReplace.empty(); }
    };

    ///\brief Remove one layer of sugar, but only some kinds.
    bool SingleStepPartiallyDesugarType(clang::QualType& QT,
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
    clang::QualType
    GetPartiallyDesugaredType(const clang::ASTContext& Ctx, clang::QualType QT,
                              const Config& TypeConfig,
                              bool fullyQualify = true);

  }

  ///\brief Class containing static utility functions looking up names. Very
  /// useful for quick, simple lookups.
  ///
  namespace Lookup {

    ///\brief Quick lookup for a single namespace declaration in a given
    /// declaration context.
    ///
    ///\param[in] S - Semantic Analysis object doing the lookup.
    ///\param[in] Name - The name we are looking up.
    ///\param[in] Within - The context within the lookup is done. If 0 the
    ///                    TranslationUnitDecl is used.
    ///\returns the found NamespaceDecl or 0.
    ///
    clang::NamespaceDecl* Namespace(clang::Sema* S,
                                    const char* Name,
                                    const clang::DeclContext* Within = 0);

    ///\brief Quick lookup for a single named declaration in a given
    /// declaration context.
    ///
    ///\param[in] S - Semantic Analysis object doing the lookup.
    ///\param[in] Name - The name we are looking up.
    ///\param[in] Within - The context within the lookup is done. If 0 the
    ///                    TranslationUnitDecl is used.
    ///\returns the found result if single, -1 if multiple or 0 if not found.
    ///
    clang::NamedDecl* Named(clang::Sema* S,
                            llvm::StringRef Name,
                            const clang::DeclContext* Within = 0);

    ///\brief Quick lookup for a single named declaration in a given
    /// declaration context.
    ///
    ///\param[in] S - Semantic Analysis object doing the lookup.
    ///\param[in] Name - The name we are looking up.
    ///\param[in] Within - The context within the lookup is done. If 0 the
    ///                    TranslationUnitDecl is used.
    ///\returns the found result if single, -1 if multiple or 0 if not found.
    ///
    clang::NamedDecl* Named(clang::Sema* S,
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
    ///\returns the found result if single, -1 if multiple or 0 if not found.
    ///
    clang::NamedDecl* Named(clang::Sema* S,
                            const clang::DeclarationName& Name,
                            const clang::DeclContext* Within = 0);

  }

  namespace TypeName {
    ///\brief Convert the type into one with fully qualified template
    /// arguments.
    ///\param[in] QT - the type for which the fully qualified type will be
    /// returned.
    ///\param[in] Ctx - the ASTContext to be used.
    clang::QualType GetFullyQualifiedType(clang::QualType QT,
                                          const clang::ASTContext& Ctx);

    ///\brief Get the fully qualified name for a type. This includes full
    /// qualification of all template parameters etc.
    ///
    ///\param[in] QT - the type for which the fully qualified name will be
    /// returned.
    ///\param[in] Ctx - the ASTContext to be used.
    std::string GetFullyQualifiedName(clang::QualType QT,
                                      const clang::ASTContext &Ctx);

    ///\brief Create a NestedNameSpecifier for Namesp and its enclosing
    /// scopes.
    ///
    ///\param[in] Ctx - the AST Context to be used.
    ///\param[in] Namesp - the NamespaceDecl for which a NestedNameSpecifier
    /// is requested.
    clang::NestedNameSpecifier*
    CreateNestedNameSpecifier(const clang::ASTContext& Ctx,
                              const clang::NamespaceDecl* Namesp);

    ///\brief Create a NestedNameSpecifier for TagDecl and its enclosing
    /// scopes.
    ///
    ///\param[in] Ctx - the AST Context to be used.
    ///\param[in] TD - the TagDecl for which a NestedNameSpecifier is
    /// requested.
    ///\param[in] FullyQualify - Convert all template arguments into fully
    /// qualified names.
    clang::NestedNameSpecifier*
    CreateNestedNameSpecifier(const clang::ASTContext& Ctx,
                              const clang::TagDecl *TD, bool FullyQualify);

    ///\brief Create a NestedNameSpecifier for TypedefDecl and its enclosing
    /// scopes.
    ///
    ///\param[in] Ctx - the AST Context to be used.
    ///\param[in] TD - the TypedefDecl for which a NestedNameSpecifier is
    /// requested.
    ///\param[in] FullyQualify - Convert all template arguments (of possible
    /// parent scopes) into fully qualified names.
    clang::NestedNameSpecifier*
    CreateNestedNameSpecifier(const clang::ASTContext& Ctx,
                              const clang::TypedefNameDecl *TD,
                              bool FullyQualify);

  } // end namespace TypeName
} // end namespace utils
} // end namespace cling
#endif // CLING_UTILS_AST_H
