//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ASTNodeEraser.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DependentDiagnostic.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"

using namespace clang;

namespace cling {

  ///\brief The class does the actual work of removing a declaration and
  /// resetting the internal structures of the compiler
  ///
  class DeclReverter : public DeclVisitor<DeclReverter, bool> {
  private:
    typedef llvm::DenseSet<FileID> FileIDs;

    ///\brief The Sema object being reverted (contains the AST as well).
    ///
    Sema* m_Sema;

    ///\brief The current transaction being reverted.
    ///
    const Transaction* m_CurTransaction;

    ///\brief The mangler used to get the mangled names of the declarations
    /// that we are removing from the module.
    ///
    llvm::OwningPtr<MangleContext> m_Mangler;


    ///\brief Reverted declaration contains a SourceLocation, representing a 
    /// place in the file where it was seen. Clang caches that file and even if
    /// a declaration is removed and the file is edited we hit the cached entry.
    /// This ADT keeps track of the files from which the reverted declarations
    /// came from so that in the end they could be removed from clang's cache.
    ///
    FileIDs m_FilesToUncache;

  public:
    DeclReverter(Sema* S, const Transaction* T): m_Sema(S), m_CurTransaction(T) {
      m_Mangler.reset(m_Sema->getASTContext().createMangleContext());
    }
    ~DeclReverter();

    ///\brief Interface with nice name, forwarding to Visit.
    ///
    ///\param[in] D - The declaration to forward.
    ///\returns true on success.
    ///
    bool RevertDecl(Decl* D) { return Visit(D); }

    ///\brief Function that contains common actions, done for every removal of
    /// declaration.
    ///
    /// For example: We must uncache the cached include, which brought that
    /// declaration in the AST.
    ///\param[in] D - A declaration.
    ///
    void PreVisitDecl(Decl* D);

    ///\brief If it falls back in the base class just remove the declaration
    /// only from the declaration context.
    /// @param[in] D - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitDecl(Decl* D);

    ///\brief Removes the declaration from the lookup chains and from the
    /// declaration context.
    /// @param[in] ND - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitNamedDecl(NamedDecl* ND);

    ///\brief Removes the declaration from the lookup chains and from the
    /// declaration context and it rebuilds the redeclaration chain.
    /// @param[in] VD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitVarDecl(VarDecl* VD);

    ///\brief Removes the declaration from the lookup chains and from the
    /// declaration context and it rebuilds the redeclaration chain.
    /// @param[in] FD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitFunctionDecl(FunctionDecl* FD);

    ///\brief Removes the enumerator and its enumerator constants.
    /// @param[in] ED - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitEnumDecl(EnumDecl* ED);


    ///\brief Removes the namespace.
    /// @param[in] NSD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitNamespaceDecl(NamespaceDecl* NSD);

    /// @name Helpers
    /// @{

    ///\brief Checks whether the declaration was pushed onto the declaration
    /// chains.
    /// @param[in] ND - The declaration that is being checked.
    ///
    ///\returns true if the ND was found in the lookup chain.
    ///
    bool isOnScopeChains(clang::NamedDecl* ND);

    ///\brief Removes given declaration from the chain of redeclarations.
    /// Rebuilds the chain and sets properly first and last redeclaration.
    /// @param[in] R - The redeclarable, its chain to be rebuilt.
    ///
    ///\returns the most recent redeclaration in the new chain.
    ///
    template <typename T>
    T* RemoveFromRedeclChain(clang::Redeclarable<T>* R) {
      llvm::SmallVector<T*, 4> PrevDecls;
      T* PrevDecl = 0;

      // [0]=>C [1]=>B [2]=>A ...
      while ((PrevDecl = R->getPreviousDecl())) {
        PrevDecls.push_back(PrevDecl);
        R = PrevDecl;
      }

      if (!PrevDecls.empty()) {
        // Put 0 in the end of the array so that the loop will reset the
        // pointer to latest redeclaration in the chain to itself.
        //
        PrevDecls.push_back(0);

        // 0 <- A <- B <- C
        for(unsigned i = PrevDecls.size() - 1; i > 0; --i) {
          PrevDecls[i-1]->setPreviousDeclaration(PrevDecls[i]);
        }
      }

      return PrevDecls.empty() ? 0 : PrevDecls[0]->getMostRecentDecl();
    }

    /// @}
  };

  DeclReverter::~DeclReverter() {
    SourceManager& SM = m_Sema->getSourceManager();
    for (FileIDs::iterator I = m_FilesToUncache.begin(), 
           E = m_FilesToUncache.end(); I != E; ++I) {
      const SrcMgr::FileInfo& fInfo = SM.getSLocEntry(*I).getFile();
      // We need to reset the cache
      SrcMgr::ContentCache* cache 
        = const_cast<SrcMgr::ContentCache*>(fInfo.getContentCache());
      FileEntry* entry = const_cast<FileEntry*>(cache->ContentsEntry);
      // We have to reset the file entry size to keep the cache and the file
      // entry in sync.
      if (entry) {
        cache->replaceBuffer(0,/*free*/true);
        FileManager::modifyFileEntry(entry, /*size*/0, 0);
      }
    }

    // Clean up the pending instantiations
    m_Sema->PendingInstantiations.clear();
    m_Sema->PendingLocalImplicitInstantiations.clear();
  }

  void DeclReverter::PreVisitDecl(Decl *D) {
    const SourceLocation Loc = D->getLocStart();
    const SourceManager& SM = m_Sema->getSourceManager();
    FileID FID = SM.getFileID(SM.getSpellingLoc(Loc));
    if (!FID.isInvalid() && !m_FilesToUncache.count(FID)) 
      m_FilesToUncache.insert(FID);
  }

  // Gives us access to the protected members that we need.
  class DeclContextExt : public DeclContext {
  public:
    static bool removeIfLast(DeclContext* DC, Decl* D) {
      if (!D->getNextDeclInContext()) {
        // Either last (remove!), or invalid (nothing to remove)
        if (((DeclContextExt*)DC)->LastDecl == D) {
          // Valid. Thus remove.
          DC->removeDecl(D);
          return true;
        }
      }
      else {
        DC->removeDecl(D);
        return true;
      }

      return false;
    }
  };

  bool DeclReverter::VisitDecl(Decl* D) {
    assert(D && "The Decl is null");
    PreVisitDecl(D);

    DeclContext* DC = D->getLexicalDeclContext();

    bool ExistsInDC = false;

    for (DeclContext::decl_iterator I = DC->decls_begin(), E = DC->decls_end();
         E !=I; ++I) {
      if (*I == D) {
        ExistsInDC = true;
        break;
      }
    }

    bool Successful = DeclContextExt::removeIfLast(DC, D);

    // ExistsInDC && Successful
    // true          false      -> false // In the context but cannot delete
    // false         false      -> true  // Not in the context cannot delete
    // true          true       -> true  // In the context and can delete
    // false         true       -> assert // Not in the context but can delete ?
    assert(!(!ExistsInDC && Successful) && \
           "Not in the context but can delete?!");
    if (ExistsInDC && !Successful)
      return false;
    else { // in release we'd want the assert to fall into true
      m_Sema->getDiagnostics().Reset();
      return true;
    }
  }

  bool DeclReverter::VisitNamedDecl(NamedDecl* ND) {
    bool Successful = VisitDecl(ND);

    DeclContext* DC = ND->getDeclContext();

    // If the decl was removed make sure that we fix the lookup
    if (Successful) {
      Scope* S = m_Sema->getScopeForContext(DC);
      if (S)
        S->RemoveDecl(ND);

      if (isOnScopeChains(ND))
        m_Sema->IdResolver.RemoveDecl(ND);

    }

    // if it was successfully removed from the AST we have to check whether
    // code was generated and remove it.
    if (Successful && m_CurTransaction->getState() == Transaction::kCommitted) {
      std::string mangledName = ND->getName();

      if (m_Mangler->shouldMangleDeclName(ND)) {
        mangledName = "";
        llvm::raw_string_ostream RawStr(mangledName);
        m_Mangler->mangleName(ND, RawStr);
        RawStr.flush();
      }

      llvm::GlobalValue* GV 
        = m_CurTransaction->getModule()->getNamedValue(mangledName);

      if (!GV->use_empty())
        GV->replaceAllUsesWith(llvm::UndefValue::get(GV->getType()));
      GV->eraseFromParent();
    }

    return Successful;
  }

  bool DeclReverter::VisitVarDecl(VarDecl* VD) {
    bool Successful = VisitNamedDecl(VD);

    DeclContext* DC = VD->getDeclContext();
    Scope* S = m_Sema->getScopeForContext(DC);

    // Find other decls that the old one has replaced
    StoredDeclsMap *Map = DC->getPrimaryContext()->getLookupPtr();
    if (!Map)
      return false;
    StoredDeclsMap::iterator Pos = Map->find(VD->getDeclName());
    // FIXME: All of that should be moved in VisitNamedDecl
    assert((VD->isHidden() || Pos != Map->end())
           && "no lookup entry for decl");

    if (Pos->second.isNull())
      // We need to rewire the list of the redeclarations in order to exclude
      // the reverted one, because it gets found for example by
      // Sema::MergeVarDecl and ends up in the lookup
      //
      if (VarDecl* MostRecentVD = RemoveFromRedeclChain(VD)) {

        Pos->second.setOnlyValue(MostRecentVD);
        if (S)
          S->AddDecl(MostRecentVD);
        m_Sema->IdResolver.AddDecl(MostRecentVD);
      }

    return Successful;
  }

  bool DeclReverter::VisitFunctionDecl(FunctionDecl* FD) {
    bool Successful = true;

    DeclContext* DC = FD->getDeclContext();
    Scope* S = m_Sema->getScopeForContext(DC);

    // Template instantiation of templated function first creates a canonical
    // declaration and after the actual template specialization. For example:
    // template<typename T> T TemplatedF(T t);
    // template<> int TemplatedF(int i) { return i + 1; } creates:
    // 1. Canonical decl: int TemplatedF(int i);
    // 2. int TemplatedF(int i){ return i + 1; }
    //
    // The template specialization is attached to the list of specialization of
    // the templated function.
    // When TemplatedF is looked up it finds the templated function and the
    // lookup is extended by the templated function with its specializations.
    // In the end we don't need to remove the canonical decl because, it
    // doesn't end up in the lookup table.
    //
    class FunctionTemplateDeclExt : public FunctionTemplateDecl {
    public:
      static void removeSpecialization(FunctionTemplateDecl* self, 
                                const FunctionTemplateSpecializationInfo* info) {
        assert(self && "Cannot be null!");
        typedef llvm::SmallVector<FunctionTemplateSpecializationInfo*, 4> FTSI;
        FunctionTemplateDeclExt* This = (FunctionTemplateDeclExt*) self;
        // We can't just copy because in the end of the scope we will call the
        // dtor of the elements.
        FunctionTemplateSpecializationInfo* specInfos
          = &(*This->getSpecializations().begin());
        size_t specInfoSize = This->getSpecializations().size();
        
        This->getSpecializations().clear();
        for (size_t i = 0; i < specInfoSize; ++i)
          if (&specInfos[i] != info) {
            This->addSpecialization(&specInfos[i], /*InsertPos*/(void*)0);
          }
      }
    };

    if (FD->isFunctionTemplateSpecialization()) {
      // 1. Remove the canonical decl.
      // TODO: Can the canonical have another DeclContext and Scope, different
      // from the specialization's implementation?
      FunctionDecl* CanFD = FD->getCanonicalDecl();
      FunctionTemplateDecl* FTD
        = FD->getTemplateSpecializationInfo()->getTemplate();
      FunctionTemplateDeclExt::removeSpecialization(FTD, 
                                         CanFD->getTemplateSpecializationInfo());
    }

    // Find other decls that the old one has replaced
    StoredDeclsMap *Map = DC->getPrimaryContext()->getLookupPtr();
    if (!Map)
      return false;
    StoredDeclsMap::iterator Pos = Map->find(FD->getDeclName());
    assert(Pos != Map->end() && "no lookup entry for decl");

    if (Pos->second.getAsDecl()) {
      Successful = VisitNamedDecl(FD) && Successful;

      Pos = Map->find(FD->getDeclName());
      assert(Pos != Map->end() && "no lookup entry for decl");

      if (Pos->second.isNull()) {
        // When we have template specialization we have to clean up
        if (FD->isFunctionTemplateSpecialization()) {
          while ((FD = FD->getPreviousDecl())) {
            Successful = VisitNamedDecl(FD) && Successful;
          }
          return true;
        }

        // We need to rewire the list of the redeclarations in order to exclude
        // the reverted one, because it gets found for example by
        // Sema::MergeVarDecl and ends up in the lookup
        //
        if (FunctionDecl* MostRecentFD = RemoveFromRedeclChain(FD)) {
          Pos->second.setOnlyValue(MostRecentFD);
          if (S)
            S->AddDecl(MostRecentFD);
          m_Sema->IdResolver.AddDecl(MostRecentFD);
        }
      }
    }
    else if (Pos->second.getAsVector()) {
      llvm::SmallVector<NamedDecl*, 4>& Decls = *Pos->second.getAsVector();
      for(llvm::SmallVector<NamedDecl*, 4>::reverse_iterator I = Decls.rbegin();
          I != Decls.rend(); ++I)
        if ((*I) == FD)
          if (FunctionDecl* MostRecentFD = RemoveFromRedeclChain(FD)) {
            // This will delete the decl from the vector, because it is 
            // generated from the decl context.
            Successful = VisitNamedDecl((*I)) && Successful;
            (*I) = MostRecentFD;
          }
    } else {
      // There are no decls. But does this really mean "unsuccessful"?
      return false;
    }

    return Successful;
  }

  bool DeclReverter::VisitEnumDecl(EnumDecl* ED) {
    bool Successful = true;

    for (EnumDecl::enumerator_iterator I = ED->enumerator_begin(),
           E = ED->enumerator_end(); I != E; ++I) {
      assert(I->getDeclName() && "EnumConstantDecl with no name?");
      Successful = VisitNamedDecl(*I) && Successful;
    }

    Successful = VisitNamedDecl(ED) && Successful;

    return Successful;
  }

  bool DeclReverter::VisitNamespaceDecl(NamespaceDecl* NSD) {
    bool Successful = VisitNamedDecl(NSD);

    //DeclContext* DC = NSD->getPrimaryContext();
    DeclContext* DC = NSD->getDeclContext();
    Scope* S = m_Sema->getScopeForContext(DC);

    // Find other decls that the old one has replaced
    StoredDeclsMap *Map = DC->getPrimaryContext()->getLookupPtr();
    if (!Map)
      return false;
    StoredDeclsMap::iterator Pos = Map->find(NSD->getDeclName());
    assert(Pos != Map->end() && "no lookup entry for decl");

    if (Pos->second.isNull())
      if (NSD != NSD->getOriginalNamespace()) {
        NamespaceDecl* NewNSD = NSD->getOriginalNamespace();
        Pos->second.setOnlyValue(NewNSD);
        if (S)
          S->AddDecl(NewNSD);
        m_Sema->IdResolver.AddDecl(NewNSD);
      }

    return Successful;
  }

  // See Sema::PushOnScopeChains
  bool DeclReverter::isOnScopeChains(NamedDecl* ND) {

    // Named decls without name shouldn't be in. Eg: struct {int a};
    if (!ND->getDeclName())
      return false;

    // Out-of-line definitions shouldn't be pushed into scope in C++.
    // Out-of-line variable and function definitions shouldn't even in C.
    if ((isa<VarDecl>(ND) || isa<FunctionDecl>(ND)) && ND->isOutOfLine() &&
        !ND->getDeclContext()->getRedeclContext()->Equals(
                        ND->getLexicalDeclContext()->getRedeclContext()))
      return false;

    // Template instantiations should also not be pushed into scope.
    if (isa<FunctionDecl>(ND) &&
        cast<FunctionDecl>(ND)->isFunctionTemplateSpecialization())
      return false; 

    // Using directives are not registered onto the scope chain
    if (isa<UsingDirectiveDecl>(ND))
      return false;

    IdentifierResolver::iterator
      IDRi = m_Sema->IdResolver.begin(ND->getDeclName()),
      IDRiEnd = m_Sema->IdResolver.end();

    for (; IDRi != IDRiEnd; ++IDRi) {
      if (ND == *IDRi)
        return true;
    }


    // Check if the declaration is template instantiation, which is not in
    // any DeclContext yet, because it came from
    // Sema::PerformPendingInstantiations
    // if (isa<FunctionDecl>(D) &&
    //     cast<FunctionDecl>(D)->getTemplateInstantiationPattern())
    //   return false;ye


    return false;
  }

  ASTNodeEraser::ASTNodeEraser(Sema* S) : m_Sema(S) {
  }

  ASTNodeEraser::~ASTNodeEraser() {
  }

  bool ASTNodeEraser::RevertTransaction(const Transaction* T) {
    DeclReverter DeclRev(m_Sema, T);
    bool Successful = true;

    for (Transaction::const_reverse_iterator I = T->rdecls_begin(),
           E = T->rdecls_end(); I != E; ++I) {
      const DeclGroupRef& DGR = (*I).m_DGR;

      for (DeclGroupRef::const_iterator
             Di = DGR.end() - 1, E = DGR.begin() - 1; Di != E; --Di) {
        // Get rid of the declaration. If the declaration has name we should
        // heal the lookup tables as well
        Successful = DeclRev.RevertDecl(*Di) && Successful;
#ifndef NDEBUG
        assert(Successful && "Cannot handle that yet!");
#endif
      }
    }

    return Successful;
  }
} // end namespace cling
