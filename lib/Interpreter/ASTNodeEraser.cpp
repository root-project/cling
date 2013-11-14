//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#include "ASTNodeEraser.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DependentDiagnostic.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JIT.h" // For debugging the EE in gdb
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/IPO.h"

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

    ///\brief The execution engine, either JIT or MCJIT, being recovered.
    ///
    llvm::ExecutionEngine* m_EEngine;

    ///\brief The current transaction being reverted.
    ///
    const Transaction* m_CurTransaction;

    ///\brief Reverted declaration contains a SourceLocation, representing a 
    /// place in the file where it was seen. Clang caches that file and even if
    /// a declaration is removed and the file is edited we hit the cached entry.
    /// This ADT keeps track of the files from which the reverted declarations
    /// came from so that in the end they could be removed from clang's cache.
    ///
    FileIDs m_FilesToUncache;

  public:
    DeclReverter(Sema* S, llvm::ExecutionEngine* EE, const Transaction* T)
      : m_Sema(S), m_EEngine(EE), m_CurTransaction(T) { }
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

    ///\brief Specialize the removal of constructors due to the fact the we need
    /// the constructor type (aka CXXCtorType). The information is located in
    /// the CXXConstructExpr of usually VarDecls. 
    /// See clang::CodeGen::CodeGenFunction::EmitCXXConstructExpr.
    ///
    /// What we will do instead is to brute-force and try to remove from the 
    /// llvm::Module all ctors of this class with all the types.
    ///
    ///\param[in] CXXCtor - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitCXXConstructorDecl(CXXConstructorDecl* CXXCtor);

    ///\brief Removes the DeclCotnext and its decls.
    /// @param[in] DC - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitDeclContext(DeclContext* DC);

    ///\brief Removes the namespace.
    /// @param[in] NSD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitNamespaceDecl(NamespaceDecl* NSD);

    ///\brief Removes a Tag (class/union/struct/enum). Most of the other
    /// containers fall back into that case.
    /// @param[in] TD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitTagDecl(TagDecl* TD);

    void RemoveDeclFromModule(GlobalDecl& GD) const;
    void RemoveStaticInit(llvm::Function& F) const;

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
    /// @param[in] DC - Remove the redecl's lookup entry from this DeclContext.
    ///
    ///\returns the most recent redeclaration in the new chain.
    ///
    template <typename T>
    bool VisitRedeclarable(clang::Redeclarable<T>* R, DeclContext* DC) {
      llvm::SmallVector<T*, 4> PrevDecls;
      T* PrevDecl = R->getMostRecentDecl();
      // [0]=>C [1]=>B [2]=>A ...
      while (PrevDecl) { // Collect the redeclarations, except the one we remove
        if (PrevDecl != R)
          PrevDecls.push_back(PrevDecl);
        PrevDecl = PrevDecl->getPreviousDecl();
      }

      if (!PrevDecls.empty()) {
        // Make sure we update the lookup maps, because the removed decl might
        // be registered in the lookup and again findable.
        StoredDeclsMap* Map = DC->getPrimaryContext()->getLookupPtr();
        if (Map) {
          NamedDecl* ND = (NamedDecl*)((T*)R);
          DeclarationName Name = ND->getDeclName();
          if (!Name.isEmpty()) {
            StoredDeclsMap::iterator Pos = Map->find(Name);
            if (Pos != Map->end() && !Pos->second.isNull()) {
              // If this is a redeclaration of an existing decl, replace the
              // old one with D.
              if (!Pos->second.HandleRedeclaration(PrevDecls[0])) {
                // We are probably in the case where we had overloads and we 
                // deleted an overload definition but we still have its 
                // declaration. Say void f(); void f(int); void f(int) {}
                // If f(int) was in the lookup table we remove it but we must
                // put the declaration of void f(int);
                if (Pos->second.getAsDecl() == ND)
                  Pos->second.setOnlyValue(PrevDecls[0]);
                else if (StoredDeclsList::DeclsTy* Vec 
                         = Pos->second.getAsVector()) {
                  bool wasReplaced = false;
                  for (StoredDeclsList::DeclsTy::iterator I= Vec->begin(),
                         E = Vec->end(); I != E; ++I)
                    if (*I == ND) {
                      // We need to replace it exactly at the same place where
                      // the old one was. The reason is cling diff-based 
                      // test suite
                      *I = PrevDecls[0];
                      wasReplaced = true;
                      break;
                    }
                  // This will make the DeclContext::removeDecl happy. It also
                  // tries to remove the decl from the lookup.
                  if (wasReplaced)
                    Pos->second.AddSubsequentDecl(ND);
                }
              }
            }
          }
        }
        // Put 0 in the end of the array so that the loop will reset the
        // pointer to latest redeclaration in the chain to itself.
        //
        PrevDecls.push_back(0);

        // 0 <- A <- B <- C
        for(unsigned i = PrevDecls.size() - 1; i > 0; --i) {
          PrevDecls[i-1]->setPreviousDeclaration(PrevDecls[i]);
        }
      }
      return true;
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
          // Force rebuilding of the lookup table.
          //DC->setMustBuildLookupTable();
          return true;
        }
      }
      else {
        // Force rebuilding of the lookup table.
        //DC->setMustBuildLookupTable();
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

#ifndef NDEBUG
    bool ExistsInDC = false;
    // The decl should be already in, we shouldn't deserialize.
    for (DeclContext::decl_iterator I = DC->noload_decls_begin(), 
           E = DC->noload_decls_end(); E !=I; ++I)
      if (*I == D) {
        ExistsInDC = true;
        break;
      }
    assert((D->isInvalidDecl() || ExistsInDC)
           && "Declaration must exist in the DC");
#endif
    bool Successful = true;
    DeclContextExt::removeIfLast(DC, D);

    // With the bump allocator this is nop.
    if (Successful)
      m_Sema->getASTContext().Deallocate(D);
    return Successful;
  }

  bool DeclReverter::VisitNamedDecl(NamedDecl* ND) {
    bool Successful = VisitDecl(ND);

    DeclContext* DC = ND->getDeclContext();

    // if the decl was anonymous we are done.
    if (!ND->getIdentifier())
      return Successful;

     // If the decl was removed make sure that we fix the lookup
    if (Successful) {
      Scope* S = m_Sema->getScopeForContext(DC);
      if (S)
        S->RemoveDecl(ND);

      if (isOnScopeChains(ND))
        m_Sema->IdResolver.RemoveDecl(ND);
    }
    
    // Cleanup the lookup tables.
    StoredDeclsMap *Map = DC->getPrimaryContext()->getLookupPtr();
    if (Map) { // DeclContexts like EnumDecls don't have lookup maps.
      // Make sure we the decl doesn't exist in the lookup tables.
      StoredDeclsMap::iterator Pos = Map->find(ND->getDeclName());
      if ( Pos != Map->end()) {
        // Most decls only have one entry in their list, special case it.
        if (Pos->second.getAsDecl() == ND)
          Pos->second.remove(ND);
        else if (StoredDeclsList::DeclsTy* Vec = Pos->second.getAsVector()) {
          // Otherwise iterate over the list with entries with the same name.
          // TODO: Walk the redeclaration chain if the entry was a redeclaration.    
          for (StoredDeclsList::DeclsTy::const_iterator I = Vec->begin(), 
                 E = Vec->end(); I != E; ++I)
            if (*I == ND)
              Pos->second.remove(ND);            
        }
        if (Pos->second.isNull())
          Map->erase(Pos);
      }
    }

#ifndef NDEBUG
    if (Map) { // DeclContexts like EnumDecls don't have lookup maps.
      // Make sure we the decl doesn't exist in the lookup tables.
      StoredDeclsMap::iterator Pos = Map->find(ND->getDeclName());
      if ( Pos != Map->end()) {
        // Most decls only have one entry in their list, special case it.
        if (NamedDecl *OldD = Pos->second.getAsDecl())
          assert(OldD != ND && "Lookup entry still exists.");
        else if (StoredDeclsList::DeclsTy* Vec = Pos->second.getAsVector()) {
          // Otherwise iterate over the list with entries with the same name.
          // TODO: Walk the redeclaration chain if the entry was a redeclaration. 
          
          for (StoredDeclsList::DeclsTy::const_iterator I = Vec->begin(), 
                 E = Vec->end(); I != E; ++I)
            assert(*I != ND && "Lookup entry still exists.");
        }
        else
          assert(Pos->second.isNull() && "!?");
      }
    }
#endif

    return Successful;
  }

  bool DeclReverter::VisitVarDecl(VarDecl* VD) {
    // VarDecl : DeclaratiorDecl, Redeclarable
    bool Successful = VisitRedeclarable(VD, VD->getDeclContext());
    Successful &= VisitDeclaratorDecl(VD);

    //If the transaction was committed we need to cleanup the execution engine.
    GlobalDecl GD(VD);
    RemoveDeclFromModule(GD);
    return Successful;
  }

  bool DeclReverter::VisitFunctionDecl(FunctionDecl* FD) {
    // FunctionDecl : DeclaratiorDecl, DeclContext, Redeclarable
    bool Successful = VisitRedeclarable(FD, FD->getDeclContext());
    Successful &= VisitDeclContext(FD);
    Successful &= VisitDeclaratorDecl(FD);

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
        void* InsertPos = 0;
        for (size_t i = 0; i < specInfoSize; ++i)
          if (&specInfos[i] != info) {
            This->addSpecialization(&specInfos[i], InsertPos);
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


    // The Structors need to be handled differently.
    if (isa<CXXConstructorDecl>(FD) || isa<CXXDestructorDecl>(FD))
      return Successful;
    //If the transaction was committed we need to cleanup the execution engine.

    GlobalDecl GD(FD);
    RemoveDeclFromModule(GD);

    return Successful;
  }

  bool DeclReverter::VisitCXXConstructorDecl(CXXConstructorDecl* CXXCtor) {
    bool Successful = VisitCXXMethodDecl(CXXCtor);

    // Brute-force all possibly generated ctors.
    // Ctor_Complete            Complete object ctor.
    // Ctor_Base                Base object ctor.
    // Ctor_CompleteAllocating 	Complete object allocating ctor. 
    GlobalDecl GD(CXXCtor, Ctor_Complete);
    RemoveDeclFromModule(GD);
    GD = GlobalDecl(CXXCtor, Ctor_Base);
    RemoveDeclFromModule(GD);
    GD = GlobalDecl(CXXCtor, Ctor_CompleteAllocating);
    RemoveDeclFromModule(GD);
    return Successful;
  }

  bool DeclReverter::VisitDeclContext(DeclContext* DC) {
    bool Successful = true;
    typedef llvm::SmallVector<Decl*, 64> Decls;
    Decls declsToErase;
    // Removing from single-linked list invalidates the iterators.
    for (DeclContext::decl_iterator I = DC->decls_begin(); 
         I != DC->decls_end(); ++I) {
      declsToErase.push_back(*I);
    }

    for(Decls::iterator I = declsToErase.begin(), E = declsToErase.end(); 
        I != E; ++I)
      Successful = Visit(*I) && Successful;
    return Successful;
  }

  bool DeclReverter::VisitNamespaceDecl(NamespaceDecl* NSD) {
    bool Successful = VisitDeclContext(NSD);

    // If this wasn't the original namespace we need to nominate a new one and
    // store it in the lookup tables.
    DeclContext* DC = NSD->getDeclContext();
    StoredDeclsMap *Map = DC->getPrimaryContext()->getLookupPtr();
    if (!Map)
      return false;
    StoredDeclsMap::iterator Pos = Map->find(NSD->getDeclName());
    assert(Pos != Map->end() && !Pos->second.isNull() 
           && "no lookup entry for decl");

    if (NSD != NSD->getOriginalNamespace()) {
      NamespaceDecl* NewNSD = NSD->getOriginalNamespace();
      Pos->second.setOnlyValue(NewNSD);
      if (Scope* S = m_Sema->getScopeForContext(DC))
        S->AddDecl(NewNSD);
      m_Sema->IdResolver.AddDecl(NewNSD);
    }

    Successful &= VisitNamedDecl(NSD);
    return Successful;
  }

  bool DeclReverter::VisitTagDecl(TagDecl* TD) {
    bool Successful = VisitDeclContext(TD);
    Successful &= VisitTypeDecl(TD);
    return Successful;
  }

  void DeclReverter::RemoveDeclFromModule(GlobalDecl& GD) const {
    using namespace llvm;
    // if it was successfully removed from the AST we have to check whether
    // code was generated and remove it.

    // From llvm's mailing list, explanation of the RAUW'd assert:
    //
    // The problem isn't with your call to 
    // replaceAllUsesWith per se, the problem is that somebody (I would guess 
    // the JIT?) is holding it in a ValueMap.
    //
    // We used to have a problem that some parts of the code would keep a 
    // mapping like so:
    //    std::map<Value *, ...>
    // while somebody else would modify the Value* without them noticing, 
    // leading to a dangling pointer in the map. To fix that, we invented the 
    // ValueMap which puts a Use that doesn't show up in the use_iterator on 
    // the Value it holds. When the Value is erased or RAUW'd, the ValueMap is 
    // notified and in this case decides that's not okay and terminates the 
    // program.
    //
    // Probably what's happened here is that the calling function has had its 
    // code generated by the JIT, but not the callee. Thus the JIT emitted a 
    // call to a generated stub, and will do the codegen of the callee once 
    // that stub is reached. Of course, once the JIT is in this state, it holds 
    // on to the Function with a ValueMap in order to prevent things from 
    // getting out of sync.
    //
    if (m_CurTransaction->getState() == Transaction::kCommitted) {
      std::string mangledName;
      utils::Analyze::maybeMangleDeclName(GD, mangledName);

      GlobalValue* GV
        = m_CurTransaction->getModule()->getNamedValue(mangledName);
      if (GV) { // May be deferred decl and thus 0
        GV->removeDeadConstantUsers();
        if (!GV->use_empty()) {
          // Assert that if there was a use it is not coming from the explicit 
          // AST node, but from the implicitly generated functions, which ensure
          // the initialization order semantics. Such functions are:
          // _GLOBAL__I* and __cxx_global_var_init*
          // 
          // We can 'afford' to drop all the references because we know that the
          // static init functions must be called only once, and that was
          // already done.
          SmallVector<User*, 4> uses;
          
          for(llvm::Value::use_iterator I = GV->use_begin(), E = GV->use_end();
              I != E; ++I) {
            uses.push_back(*I);
          }

          for(SmallVector<User*, 4>::iterator I = uses.begin(), E = uses.end();
              I != E; ++I)
            if (llvm::Instruction* instr = dyn_cast<llvm::Instruction>(*I)) {
              llvm::Function* F = instr->getParent()->getParent();
              if (F->getName().startswith("__cxx_global_var_init"))
                RemoveStaticInit(*F);
          }
        
        }

        // Cleanup the jit mapping of GV->addr.
        m_EEngine->updateGlobalMapping(GV, 0);
        GV->dropAllReferences();
        if (!GV->use_empty()) {
          if (Function* F = dyn_cast<Function>(GV)) {
            Function* dummy = Function::Create(F->getFunctionType(), F->getLinkage());                                               
            F->replaceAllUsesWith(dummy);
          }
          else
            GV->replaceAllUsesWith(UndefValue::get(GV->getType()));
        }
        GV->eraseFromParent();
      }
    }
  }

  void DeclReverter::RemoveStaticInit(llvm::Function& F) const {
    // In our very controlled case the parent of the BasicBlock is the 
    // static init llvm::Function.
    assert(F.getName().startswith("__cxx_global_var_init")
           && "Not a static init");
    assert(F.hasInternalLinkage() && "Not a static init");
    // The static init functions have the layout:
    // declare internal void @__cxx_global_var_init1() section "..."
    //
    // define internal void @_GLOBAL__I_a2() section "..." {
    // entry:
    //  call void @__cxx_global_var_init1()
    //  ret void
    // }
    //
    assert(F.hasOneUse() && "Must have only one use");
    // erase _GLOBAL__I* first
    llvm::BasicBlock* BB = cast<llvm::Instruction>(F.use_back())->getParent();
    BB->getParent()->eraseFromParent();
    F.eraseFromParent();
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

  ASTNodeEraser::ASTNodeEraser(Sema* S, llvm::ExecutionEngine* EE)
    : m_Sema(S), m_EEngine(EE) { }

  ASTNodeEraser::~ASTNodeEraser() {
  }

  bool ASTNodeEraser::RevertTransaction(Transaction* T) {
    DeclReverter DeclRev(m_Sema, m_EEngine, T);
    bool Successful = true;

    for (Transaction::const_iterator I = T->decls_begin(),
           E = T->decls_end(); I != E; ++I) {
      if ((*I).m_Call != Transaction::kCCIHandleTopLevelDecl)
        continue;
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
    m_Sema->getDiagnostics().Reset();

    // Cleanup the module from unused global values.
    //llvm::ModulePass* globalDCE = llvm::createGlobalDCEPass();
    //globalDCE->runOnModule(*T->getModule());
    if (Successful)
      T->setState(Transaction::kRolledBack);
    else
      T->setState(Transaction::kRolledBackWithErrors);

    return Successful;
  }
} // end namespace cling
