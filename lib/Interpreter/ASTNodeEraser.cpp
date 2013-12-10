//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
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
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"

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

    ///\brief Remove the macro from the Preprocessor.
    /// @param[in] MD - The MacroDirectiveInfo containing the IdentifierInfo and
    ///                MacroDirective to forward.
    ///
    ///\returns true on success.
    ///
    bool VisitMacro(const Transaction::MacroDirectiveInfo MD);

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

    ///\brief Interface with nice name, forwarding to Visit.
    ///
    ///\param[in] MD - The MacroDirectiveInfo containing the IdentifierInfo and
    ///                MacroDirective to forward.
    ///\returns true on success.
    ///
    bool RevertMacro(const Transaction::MacroDirectiveInfo MD) { return VisitMacro(MD); }

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

  private:
    ///\brief Function that collects the files which we must reread from disk.
    ///
    /// For example: We must uncache the cached include, which brought a
    /// declaration or a macro diretive definition in the AST.
    ///\param[in] Loc - The source location of the reverted declaration.
    ///
    void CollectFilesToUncache(SourceLocation Loc);

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

  void DeclReverter::CollectFilesToUncache(SourceLocation Loc) {
    const SourceManager& SM = m_Sema->getSourceManager();
    FileID FID = SM.getFileID(SM.getSpellingLoc(Loc));
    if (!FID.isInvalid() && FID >= m_CurTransaction->getBufferFID()
        && !m_FilesToUncache.count(FID)) 
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
    CollectFilesToUncache(D->getLocStart());

    DeclContext* DC = D->getLexicalDeclContext();

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
    while (DC->isTransparentContext())
      DC = DC->getLookupParent();

    // if the decl was anonymous we are done.
    if (!ND->getIdentifier())
      return Successful;

     // If the decl was removed make sure that we fix the lookup
    if (Successful) {
      if (Scope* S = m_Sema->getScopeForContext(DC))
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
    // Cleanup the module if the transaction was committed and code was 
    // generated. This has to go first, because it may need the AST information
    // which we will remove soon. (Eg. mangleDeclName iterates the redecls)
    GlobalDecl GD(VD);
    RemoveDeclFromModule(GD);

    // VarDecl : DeclaratiorDecl, Redeclarable
    bool Successful = VisitRedeclarable(VD, VD->getDeclContext());
    Successful &= VisitDeclaratorDecl(VD);

    return Successful;
  }

  bool DeclReverter::VisitFunctionDecl(FunctionDecl* FD) {
    // The Structors need to be handled differently.
    if (!isa<CXXConstructorDecl>(FD) && !isa<CXXDestructorDecl>(FD)) {
      // Cleanup the module if the transaction was committed and code was 
      // generated. This has to go first, because it may need the AST info
      // which we will remove soon. (Eg. mangleDeclName iterates the redecls)
      GlobalDecl GD(FD);
      RemoveDeclFromModule(GD);
    }

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
                                       const FunctionDecl* specialization) {
        assert(self && specialization && "Cannot be null!");
        assert(specialization == specialization->getCanonicalDecl() 
               && "Not the canonical specialization!?");
        typedef llvm::SmallVector<FunctionDecl*, 4> Specializations;
        typedef llvm::FoldingSetVector< FunctionTemplateSpecializationInfo> Set;

        FunctionTemplateDeclExt* This = (FunctionTemplateDeclExt*) self;
        Specializations specializations;
        const Set& specs = This->getSpecializations();

        if (!specs.size()) // nothing to remove
          return;

        // Collect all the specializations without the one to remove.
        for(Set::const_iterator I = specs.begin(),E = specs.end(); I != E; ++I){
          assert(I->Function && "Must have a specialization.");
          if (I->Function != specialization)
            specializations.push_back(I->Function);
        }

        This->getSpecializations().clear();

        //Readd the collected specializations.
        void* InsertPos = 0;
        FunctionTemplateSpecializationInfo* FTSI = 0;
        for (size_t i = 0, e = specializations.size(); i < e; ++i) {
          FTSI = specializations[i]->getTemplateSpecializationInfo();
          assert(FTSI && "Must not be null.");
          // Avoid assertion on add.
          FTSI->SetNextInBucket(0);
          This->addSpecialization(FTSI, InsertPos);
        }
#ifndef NDEBUG
        const TemplateArgumentList* args
          = specialization->getTemplateSpecializationArgs();
        assert(!self->findSpecialization(args->data(), args->size(),  InsertPos)
               && "Finds the removed decl again!");
#endif
      }
    };

    if (FD->isFunctionTemplateSpecialization() && FD->isCanonicalDecl()) {
      // Only the canonical declarations are registered in the list of the
      // specializations.
      FunctionTemplateDecl* FTD
        = FD->getTemplateSpecializationInfo()->getTemplate();
      // The canonical declaration of every specialization is registered with
      // the FunctionTemplateDecl.
      // Note this might revert too much in the case:
      //   template<typename T> T f(){ return T();}
      //   template<> int f();
      //   template<> int f() { return 0;}
      // when the template specialization was forward declared the canonical
      // becomes the first forward declaration. If the canonical forward
      // declaration was declared outside the set of the decls to revert we have
      // to keep it registered as a template specialization.
      // 
      // In order to diagnose mismatches of the specializations, clang 'injects'
      // a implicit forward declaration making it very hard distinguish between
      // the explicit and the implicit forward declaration. So far the only way
      // to distinguish is by source location comparison.
      // FIXME: When the misbehavior of clang is fixed we must avoid relying on
      // source locations
      FunctionTemplateDeclExt::removeSpecialization(FTD, FD);
    }

    return Successful;
  }

  bool DeclReverter::VisitCXXConstructorDecl(CXXConstructorDecl* CXXCtor) {
    // Cleanup the module if the transaction was committed and code was 
    // generated. This has to go first, because it may need the AST information
    // which we will remove soon. (Eg. mangleDeclName iterates the redecls)

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

    bool Successful = VisitCXXMethodDecl(CXXCtor);
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

  bool DeclReverter::VisitMacro(const Transaction::MacroDirectiveInfo MacroD) {
    assert(MacroD.m_MD && "The MacroDirective is null");
    assert(MacroD.m_II && "The IdentifierInfo is null");
    CollectFilesToUncache(MacroD.m_MD->getLocation());

    Preprocessor& PP = m_Sema->getPreprocessor();
#ifndef NDEBUG
    bool ExistsInPP = false;
    // Make sure the macro is in the Preprocessor. Not sure if not redundant
    // because removeMacro looks for the macro anyway in the DenseMap Macros[]
    for (Preprocessor::macro_iterator 
           I = PP.macro_begin(/*IncludeExternalMacros*/false), 
           E = PP.macro_end(/*IncludeExternalMacros*/false); E !=I; ++I) {
      if ((*I).first == MacroD.m_II) {
        // FIXME:check whether we have the concrete directive on the macro chain
        // && (*I).second == MacroD.m_MD
        ExistsInPP = true;
        break;
      }
    }
    assert(ExistsInPP && "Not in the Preprocessor?!");
#endif

    const MacroDirective* MD = MacroD.m_MD;
    // Undef the definition
    const MacroInfo* MI = MD->getMacroInfo();

    // If the macro is not defined, this is a noop undef, just return.
    if (MI == 0)
      return false;

    // Remove the pair from the macros
    PP.removeMacro(MacroD.m_II, const_cast<MacroDirective*>(MacroD.m_MD));

    return true;
  }

  ASTNodeEraser::ASTNodeEraser(Sema* S, llvm::ExecutionEngine* EE)
    : m_Sema(S), m_EEngine(EE) {
  }

  ASTNodeEraser::~ASTNodeEraser() {
  }

  bool ASTNodeEraser::RevertTransaction(Transaction* T) {
    DeclReverter DeclRev(m_Sema, m_EEngine, T);
    bool Successful = true;

    for (Transaction::const_reverse_iterator I = T->rdecls_begin(),
           E = T->rdecls_end(); I != E; ++I) {
      if ((*I).m_Call != Transaction::kCCIHandleTopLevelDecl)
        continue;
      const DeclGroupRef& DGR = (*I).m_DGR;

      for (DeclGroupRef::const_iterator
             Di = DGR.end() - 1, E = DGR.begin() - 1; Di != E; --Di) {
        // Get rid of the declaration. If the declaration has name we should
        // heal the lookup tables as well
        Successful = DeclRev.RevertDecl(*Di) && Successful;

      }
    }

#ifndef NDEBUG
    assert(Successful && "Cannot handle that yet!");
#endif

    m_Sema->getDiagnostics().Reset();
    m_Sema->getDiagnostics().getClient()->clear();

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
