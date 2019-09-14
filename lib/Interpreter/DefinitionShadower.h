//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Javier López-Gómez <jalopezg@inf.uc3m.es>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_DEFINITION_SHADOWER
#define CLING_DEFINITION_SHADOWER

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
  class Interpreter;
}

namespace cling {
  /// \brief Enables shadowing of definitions, as per https://github.com/root-project/cling/issues/259.
  ///
  /// A top-level declaration typed in the Cling prompt is nested in a
  ///  `__cling_N5xxx' inline namespace, so that the newly provided definition
  /// doesn't clash, e.g. in unmodified Cling, \code unsigned i = 0U;
  /// double i = 1.0; \endcode fails to compile (error: redefinition of 'i' with
  /// a different type: 'double' vs 'unsigned int').  To support that, the
  /// previous code is transformed into:
  /// \code
  /// inline namespace __cling_N50 { unsigned i = 0U; }
  /// inline namespace __cling_N51 { double i = 1.0; }
  /// \endcode
  ///
  /// While this works for providing a new definition, any non-qualified lookup,
  /// i.e. using the name `i` instead of `__cling___N50::i` would be ambiguous.
  /// To allow the use of unqualified names, and therefore make this
  /// transformation transparent to the user, any previous definition that can
  /// be found from the TU scope is hidden from SemaLookup.
  ///
  /// It is still possible to reach previous definitions through the qualified
  /// name `__cling_N5xxx::yyy'.
  ///
  class DefinitionShadower : public ASTTransformer {
  private:
    clang::ASTContext          &m_Context;
    Interpreter                &m_Interp;
    clang::TranslationUnitDecl *m_TU;

    unsigned long long m_UniqueNameCounter;

    /// \brief Hide a global declaration from SemaLookup; internally used in
    /// `invalidatePreviousDefinitions()'. This directly manipulates lookup
    /// tables to avoid a patch to Clang.
    ///
    void hideDecl(clang::NamedDecl *D) const;

    /// \brief Lookup the given name and invalidate all clashing declarations
    /// (as seen from the TU).  `D' may be invalidated (if not a definition)
    /// and a definition for that declaration is in scope, e.g.
    /// \code class C {}; class C; \endcode
    ///
    void invalidatePreviousDefinitions(clang::NamedDecl *D) const;

    /// \brief Invalidate previous function definition.  If `D` is a wrapper,
    /// local declararations may be moved by DeclExtractor; in that case,
    /// invalidate all those before DeclExtractor runs.
    ///
    void invalidatePreviousDefinitions(clang::FunctionDecl *D) const;

    /// \brief Invalidate a previous decl of `D' that provide a definition.
    ///
    /// \param D[in] - Declaration whose name will be used to lookup existing
    /// definitions.
    ///
    void invalidatePreviousDefinitions(clang::Decl *D) const;

  public:
    DefinitionShadower(clang::Sema& S, Interpreter& I);

    /// \brief May transform the given declaration (see class documentation).
    /// Particularly, it may nest `D` in a `__cling_N5xxx` inline namespace and
    /// invalidate any previous definition currently in scope.
    ///
    Result Transform(clang::Decl* D) override;

    /// \brief Return whether `DC` is a `__cling_N5xxx` inline namespace used
    /// for definition shadowing.
    ///
    static bool isClingShadowNamespace(const clang::DeclContext *DC);
  };
} // end namespace cling

#endif // CLING_DEFINITION_SHADOWER
