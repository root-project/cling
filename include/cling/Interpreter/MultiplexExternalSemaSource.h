//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_MULTIPLEX_EXTERNAL_SEMA_SOURCE_H
#define CLING_MULTIPLEX_EXTERNAL_SEMA_SOURCE_H

#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Weak.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace clang {
  class CXXConstructorDecl;
  class CXXRecordDecl;
  class DeclaratorDecl;
  struct ExternalVTableUse;
  class LookupResult;
  class NamespaceDecl;
  class Scope;
  class Sema;
  class TypedefNameDecl;
  class ValueDecl;
  class VarDecl;
}

namespace cling {

/// \brief An abstract interface that should be implemented by
/// external AST sources that also provide information for semantic
/// analysis.
class MultiplexExternalSemaSource : public clang::ExternalSemaSource {

private:
  llvm::SmallVector<ExternalSemaSource*, 4> m_Sources; // doesn't own them.

public:

  ///\brief Constructs the external source with given elements.
  ///
  ///\param[in] sources - Array of ExternalSemaSources.
  ///
  MultiplexExternalSemaSource(llvm::ArrayRef<ExternalSemaSource*> sources);

  ~MultiplexExternalSemaSource();

  /// \brief Initialize the semantic source with the Sema instance
  /// being used to perform semantic analysis on the abstract syntax
  /// tree.
  virtual void InitializeSema(clang::Sema& S);

  /// \brief Inform the semantic consumer that Sema is no longer available.
  virtual void ForgetSema();

  /// \brief Load the contents of the global method pool for a given
  /// selector.
  virtual void ReadMethodPool(clang::Selector Sel);

  /// \brief Load the set of namespaces that are known to the external source,
  /// which will be used during typo correction.
  virtual void ReadKnownNamespaces(
                      clang::SmallVectorImpl<clang::NamespaceDecl*>& Namespaces);
  
  /// \brief Do last resort, unqualified lookup on a LookupResult that
  /// Sema cannot find.
  ///
  /// \param R a LookupResult that is being recovered.
  ///
  /// \param S the Scope of the identifier occurrence.
  ///
  /// \return true to tell Sema to recover using the LookupResult.
  virtual bool LookupUnqualified(clang::LookupResult& R, clang::Scope* S);

  /// \brief Read the set of tentative definitions known to the external Sema
  /// source.
  ///
  /// The external source should append its own tentative definitions to the
  /// given vector of tentative definitions. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadTentativeDefinitions(
                         clang::SmallVectorImpl<clang::VarDecl*>& TentativeDefs);
  
  /// \brief Read the set of unused file-scope declarations known to the
  /// external Sema source.
  ///
  /// The external source should append its own unused, filed-scope to the
  /// given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadUnusedFileScopedDecls(
                    clang::SmallVectorImpl<const clang::DeclaratorDecl*>& Decls);
  
  /// \brief Read the set of delegating constructors known to the
  /// external Sema source.
  ///
  /// The external source should append its own delegating constructors to the
  /// given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadDelegatingConstructors(
                      clang::SmallVectorImpl<clang::CXXConstructorDecl*>& Decls);

  /// \brief Read the set of ext_vector type declarations known to the
  /// external Sema source.
  ///
  /// The external source should append its own ext_vector type declarations to
  /// the given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadExtVectorDecls(
                         clang::SmallVectorImpl<clang::TypedefNameDecl*>& Decls);

  /// \brief Read the set of dynamic classes known to the external Sema source.
  ///
  /// The external source should append its own dynamic classes to
  /// the given vector of declarations. Note that this routine may be
  /// invoked multiple times; the external source should take care not to
  /// introduce the same declarations repeatedly.
  virtual void ReadDynamicClasses(
                           clang::SmallVectorImpl<clang::CXXRecordDecl*>& Decls);

  /// \brief Read the set of locally-scoped external declarations known to the
  /// external Sema source.
  ///
  /// The external source should append its own locally-scoped external
  /// declarations to the given vector of declarations. Note that this routine 
  /// may be invoked multiple times; the external source should take care not 
  /// to introduce the same declarations repeatedly.
  virtual void ReadLocallyScopedExternalDecls(
                               clang::SmallVectorImpl<clang::NamedDecl*>& Decls);

  /// \brief Read the set of referenced selectors known to the
  /// external Sema source.
  ///
  /// The external source should append its own referenced selectors to the 
  /// given vector of selectors. Note that this routine 
  /// may be invoked multiple times; the external source should take care not 
  /// to introduce the same selectors repeatedly.
  virtual void ReadReferencedSelectors(
                                clang::SmallVectorImpl<std::pair<clang::Selector, 
                                                 clang::SourceLocation> >& Sels);

  /// \brief Read the set of weak, undeclared identifiers known to the
  /// external Sema source.
  ///
  /// The external source should append its own weak, undeclared identifiers to
  /// the given vector. Note that this routine may be invoked multiple times; 
  /// the external source should take care not to introduce the same identifiers
  /// repeatedly.
  virtual void ReadWeakUndeclaredIdentifiers(
                        clang::SmallVectorImpl<std::pair<clang::IdentifierInfo*, 
                                                         clang::WeakInfo> >& WI);

  /// \brief Read the set of used vtables known to the external Sema source.
  ///
  /// The external source should append its own used vtables to the given
  /// vector. Note that this routine may be invoked multiple times; the external
  /// source should take care not to introduce the same vtables repeatedly.
  virtual void ReadUsedVTables(clang::SmallVectorImpl<clang::ExternalVTableUse>& VTables);

  /// \brief Read the set of pending instantiations known to the external
  /// Sema source.
  ///
  /// The external source should append its own pending instantiations to the
  /// given vector. Note that this routine may be invoked multiple times; the
  /// external source should take care not to introduce the same instantiations
  /// repeatedly.
  virtual void ReadPendingInstantiations(
                             clang::SmallVectorImpl<std::pair<clang::ValueDecl*, 
                                              clang::SourceLocation> >& Pending);

  // isa/cast/dyn_cast support
  static bool classof(const MultiplexExternalSemaSource*) { return true; }
}; 

} // end namespace clang

#endif // CLING_MULTIPLEX_EXTERNAL_SEMA_SOURCE_H
