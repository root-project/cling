//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/MultiplexExternalSemaSource.h"

#include "clang/Sema/Lookup.h"

using namespace clang;

namespace cling {
  MultiplexExternalSemaSource::MultiplexExternalSemaSource(
                                   llvm::ArrayRef<ExternalSemaSource*> sources) {
    for (size_t i = 0; i < sources.size(); ++i)
      m_Sources.push_back(sources[i]);
  }


  // pin the vtable here.
  MultiplexExternalSemaSource::~MultiplexExternalSemaSource() {}

  void MultiplexExternalSemaSource::InitializeSema(Sema& S) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->InitializeSema(S);
  }

  void MultiplexExternalSemaSource::ForgetSema() {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ForgetSema();
  }

  void MultiplexExternalSemaSource::ReadMethodPool(Selector Sel) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadMethodPool(Sel);
  }

  void MultiplexExternalSemaSource::ReadKnownNamespaces(
                                   SmallVectorImpl<NamespaceDecl*>& Namespaces){
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadKnownNamespaces(Namespaces);
  }
  
  bool MultiplexExternalSemaSource::LookupUnqualified(LookupResult& R, Scope* S){ 
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->LookupUnqualified(R, S);
    
    return !R.empty();
  }

  void MultiplexExternalSemaSource::ReadTentativeDefinitions(
                                     SmallVectorImpl<VarDecl*>& TentativeDefs) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadTentativeDefinitions(TentativeDefs);
  }
  
  void MultiplexExternalSemaSource::ReadUnusedFileScopedDecls(
                                SmallVectorImpl<const DeclaratorDecl*>& Decls) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadUnusedFileScopedDecls(Decls);
  }
  
  void MultiplexExternalSemaSource::ReadDelegatingConstructors(
                                  SmallVectorImpl<CXXConstructorDecl*>& Decls) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadDelegatingConstructors(Decls);
  }

  void MultiplexExternalSemaSource::ReadExtVectorDecls(
                                     SmallVectorImpl<TypedefNameDecl*>& Decls) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadExtVectorDecls(Decls);
  }

  void MultiplexExternalSemaSource::ReadDynamicClasses(
                                       SmallVectorImpl<CXXRecordDecl*>& Decls) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadDynamicClasses(Decls);
  }

  void MultiplexExternalSemaSource::ReadLocallyScopedExternalDecls(
                                           SmallVectorImpl<NamedDecl*>& Decls) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadLocallyScopedExternalDecls(Decls);
  }

  void MultiplexExternalSemaSource::ReadReferencedSelectors(
                  SmallVectorImpl<std::pair<Selector, SourceLocation> >& Sels) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadReferencedSelectors(Sels);
  }

  void MultiplexExternalSemaSource::ReadWeakUndeclaredIdentifiers(
                   SmallVectorImpl<std::pair<IdentifierInfo*, WeakInfo> >& WI) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadWeakUndeclaredIdentifiers(WI);
  }

  void MultiplexExternalSemaSource::ReadUsedVTables(
                                  SmallVectorImpl<ExternalVTableUse>& VTables) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadUsedVTables(VTables);
  }

  void MultiplexExternalSemaSource::ReadPendingInstantiations(
                                           SmallVectorImpl<std::pair<ValueDecl*,
                                                   SourceLocation> >& Pending) {
    for(size_t i = 0; i < m_Sources.size(); ++i)
      m_Sources[i]->ReadPendingInstantiations(Pending);
  }
} // end namespace cling
