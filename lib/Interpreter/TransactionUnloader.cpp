//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "TransactionUnloader.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DependentDiagnostic.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
//#include "llvm/Transforms/IPO.h"

using namespace clang;

// FIXME: rename back to cling when gcc fix the
// namespace cling { using cling::DeclUnloader DeclUnloader} bug
namespace clang {
  using namespace cling;

  // Copied and adapted from GlobalDCE.cpp
  class GlobalValueEraser {
  private:
    typedef llvm::SmallPtrSet<llvm::GlobalValue*, 32> Globals;
    Globals VisitedGlobals;
    llvm::SmallPtrSet<llvm::Constant *, 8> SeenConstants;
    clang::CodeGenerator* m_CodeGen;
  public:
    GlobalValueEraser(clang::CodeGenerator* CG)
      : m_CodeGen(CG) { }

    ///\brief Erases the given global value and all unused leftovers
    ///
    ///\param[in] GV - The removal starting point.
    ///
    ///\returns true if something was erased.
    ///
    bool EraseGlobalValue(llvm::GlobalValue* GV) {
      using namespace llvm;
      bool Changed = false;

      Changed |= RemoveUnusedGlobalValue(*GV);
      // Collect all uses of globals by GV
      CollectAllUsesOfGlobals(GV);
      FindUsedValues(*GV->getParent());

      // The first pass is to drop initializers of global vars which are dead.
      for (Globals::iterator I = VisitedGlobals.begin(),
             E = VisitedGlobals.end(); I != E; ++I)
        if (GlobalVariable* GV = dyn_cast<GlobalVariable>(*I)) {
          GV->setInitializer(0);
        }
        else if (GlobalAlias* GA = dyn_cast<GlobalAlias>(*I)) {
          GA->setAliasee(0);
        }
        else {
          Function* F = cast<Function>(*I);
          if (!F->isDeclaration())
            F->deleteBody();
        }

      if (!VisitedGlobals.empty()) {
        // Now that all interferences have been dropped, delete the actual
        // objects themselves.
        for (Globals::iterator I = VisitedGlobals.begin(),
               E = VisitedGlobals.end(); I != E; ++I) {
          RemoveUnusedGlobalValue(**I);
          if ((*I)->getNumUses())
            continue;

          // Required by ::DwarfEHPrepare::InsertUnwindResumeCalls (in the JIT)
          if ((*I)->getName().equals("_Unwind_Resume"))
            continue;

          m_CodeGen->forgetGlobal(*I);
          (*I)->eraseFromParent();
        }
        Changed = true;
      }

      // Make sure that all memory is released
      VisitedGlobals.clear();
      SeenConstants.clear();

      return Changed;
    }

  private:

    /// Find values that are marked as llvm.used.
    void FindUsedValues(const llvm::Module& m) {
      for (const llvm::GlobalVariable& GV : m.globals()) {
        if (!GV.getName().startswith("llvm.used"))
          continue;

        const llvm::ConstantArray* Inits
          = cast<llvm::ConstantArray>(GV.getInitializer());

        for (unsigned i = 0, e = Inits->getNumOperands(); i != e; ++i) {
          llvm::Value *Operand
            = Inits->getOperand(i)->stripPointerCastsNoFollowAliases();
          VisitedGlobals.erase(cast<llvm::GlobalValue>(Operand));
        }

      }
    }
    /// CollectAllUsesOfGlobals - collects recursively all referenced globals by
    /// GV.
    void CollectAllUsesOfGlobals(llvm::GlobalValue *G) {
      using namespace llvm;
      // If the global is already in the set, no need to reprocess it.
      if (!VisitedGlobals.insert(G).second)
        return;

      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(G)) {
        // If this is a global variable, we must make sure to add any global
        // values referenced by the initializer to the collection set.
        if (GV->hasInitializer())
          MarkConstant(GV->getInitializer());
      } else if (GlobalAlias *GA = dyn_cast<GlobalAlias>(G)) {
        // The target of a global alias as referenced.
        MarkConstant(GA->getAliasee());
      } else {
        // Otherwise this must be a function object.  We have to scan the body
        // of the function looking for constants and global values which are
        // used as operands.  Any operands of these types must be processed to
        // ensure that any globals used will be marked as collected.
        Function *F = cast<Function>(G);

        if (F->hasPrefixData())
          MarkConstant(F->getPrefixData());

        for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
          for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
            for (User::op_iterator U = I->op_begin(), E = I->op_end();U!=E; ++U)
              if (GlobalValue *GV = dyn_cast<GlobalValue>(*U))
                CollectAllUsesOfGlobals(GV);
              else if (Constant *C = dyn_cast<Constant>(*U))
                MarkConstant(C);
      }
    }

    void MarkConstant(llvm::Constant *C) {
      using namespace llvm;
      if (GlobalValue *GV = dyn_cast<GlobalValue>(C))
        return CollectAllUsesOfGlobals(GV);

      // Loop over all of the operands of the constant, adding any globals they
      // use to the list of needed globals.
      for (User::op_iterator I = C->op_begin(), E = C->op_end(); I != E; ++I) {
        Constant *Op = dyn_cast<Constant>(*I);
        // We already processed this constant there's no need to do it again.
        if (Op && SeenConstants.insert(Op).second)
          MarkConstant(Op);
      }
    }

    // RemoveUnusedGlobalValue - Loop over all of the uses of the specified
    // GlobalValue, looking for the constant pointer ref that may be pointing to
    // it. If found, check to see if the constant pointer ref is safe to
    // destroy, and if so, nuke it.  This will reduce the reference count on the
    // global value, which might make it deader.
    //
    bool RemoveUnusedGlobalValue(llvm::GlobalValue &GV) {
      using namespace llvm;
      if (GV.use_empty())
        return false;
      GV.removeDeadConstantUsers();
      return GV.use_empty();
    }
  };

  ///\brief The class does the actual work of removing a declaration and
  /// resetting the internal structures of the compiler
  ///
  class DeclUnloader : public DeclVisitor<DeclUnloader, bool> {
  private:
    typedef llvm::DenseSet<FileID> FileIDs;

    ///\brief The Sema object being unloaded (contains the AST as well).
    ///
    Sema* m_Sema;

    ///\brief The clang code generator, being recovered.
    ///
    clang::CodeGenerator* m_CodeGen;

    ///\brief The current transaction being unloaded.
    ///
    const Transaction* m_CurTransaction;

    ///\brief Unloaded declaration contains a SourceLocation, representing a
    /// place in the file where it was seen. Clang caches that file and even if
    /// a declaration is removed and the file is edited we hit the cached entry.
    /// This ADT keeps track of the files from which the unloaded declarations
    /// came from so that in the end they could be removed from clang's cache.
    ///
    FileIDs m_FilesToUncache;

  public:
    DeclUnloader(Sema* S, clang::CodeGenerator* CG, const Transaction* T)
      : m_Sema(S), m_CodeGen(CG), m_CurTransaction(T) { }
    ~DeclUnloader();

    ///\brief Interface with nice name, forwarding to Visit.
    ///
    ///\param[in] D - The declaration to forward.
    ///\returns true on success.
    ///
    bool UnloadDecl(Decl* D) { return Visit(D); }

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

    ///\brief Removes the declaration from Sema's unused decl registry
    /// @param[in] DD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitDeclaratorDecl(DeclaratorDecl* DD);

    ///\brief Removes a using shadow declaration, created in the cases:
    ///\code
    /// namespace A {
    ///   void foo();
    /// }
    /// namespace B {
    ///   using A::foo; // <- a UsingDecl
    ///                 // Also creates a UsingShadowDecl for A::foo() in B
    /// }
    ///\endcode
    ///\param[in] USD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitUsingShadowDecl(UsingShadowDecl* USD);

    ///\brief Removes a typedef name decls. A base class for TypedefDecls and
    /// TypeAliasDecls.
    ///\param[in] TND - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitTypedefNameDecl(TypedefNameDecl* TND);

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

    ///\brief Specialize the removal of destructors due to the fact the we need
    /// the to erase the dtor decl and the deleting operator.
    ///
    /// We will brute-force and try to remove from the llvm::Module all dtors of
    /// this class with all the types.
    ///
    ///\param[in] CXXDtor - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitCXXDestructorDecl(CXXDestructorDecl* CXXDtor);

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

    ///\brief Removes a RecordDecl. We shouldn't remove the implicit class
    /// declaration.
    ///\param[in] RD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitRecordDecl(RecordDecl* RD);

    ///\brief Remove the macro from the Preprocessor.
    /// @param[in] MD - The MacroDirectiveInfo containing the IdentifierInfo and
    ///                MacroDirective to forward.
    ///
    ///\returns true on success.
    ///
    bool VisitMacro(const Transaction::MacroDirectiveInfo MD);

    ///@name Templates
    ///@{

    ///\brief Removes template from the redecl chain. Templates are
    /// redeclarables also.
    /// @param[in] R - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl* R);


    ///\brief Removes the declaration clang's internal structures. This case
    /// looks very much to VisitFunctionDecl, but FunctionTemplateDecl doesn't
    /// derive from FunctionDecl and thus we need to handle it 'by hand'.
    /// @param[in] FTD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitFunctionTemplateDecl(FunctionTemplateDecl* FTD);

    ///\brief Removes a class template declaration from clang's internal
    /// structures.
    /// @param[in] CTD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitClassTemplateDecl(ClassTemplateDecl* CTD);

    ///\brief Removes a class template specialization declaration from clang's
    /// internal structures.
    /// @param[in] CTSD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitClassTemplateSpecializationDecl(ClassTemplateSpecializationDecl*
                                              CTSD);

    ///@}

    void MaybeRemoveDeclFromModule(GlobalDecl& GD) const;

    /// @name Helpers
    /// @{

    ///\brief Interface with nice name, forwarding to Visit.
    ///
    ///\param[in] MD - The MacroDirectiveInfo containing the IdentifierInfo and
    ///                MacroDirective to forward.
    ///\returns true on success.
    ///
    bool UnloadMacro(Transaction::MacroDirectiveInfo MD) {
      return VisitMacro(MD);
    }

    template <typename T>
    constexpr static bool isDefinition(T*) {
      return false;
    }
    static bool isDefinition(TagDecl* R) {
      return R->isCompleteDefinition() && dyn_cast<CXXRecordDecl>(R);
    }
    template <typename T>
    static void resetDefinitionData(T*) {
      llvm_unreachable("resetDefinitionData on non-cxx record declaration");
    }
    static void resetDefinitionData(TagDecl *decl) {
      auto canon = dyn_cast<CXXRecordDecl>(decl->getCanonicalDecl());
      assert(canon && "Only CXXRecordDecl have DefinitionData");
      for (auto iter = canon->getMostRecentDecl(); iter;
           iter = iter->getPreviousDecl()) {
        auto declcxx = dyn_cast<CXXRecordDecl>(iter);
        assert(declcxx && "Only CXXRecordDecl have DefinitionData");
        declcxx->DefinitionData = canon;
      }
    }

    // Copied and adapted from: ASTReaderDecl.cpp
    template<typename DeclT>
    void removeRedeclFromChain(DeclT* R) {
      //RedeclLink is a protected member.
      struct RedeclDerived : public Redeclarable<DeclT> {
        typedef typename Redeclarable<DeclT>::DeclLink DeclLink;
        static DeclLink& getLink(DeclT* R) {
          Redeclarable<DeclT>* D = R;
          return ((RedeclDerived*)D)->RedeclLink;
        }
        static void setLatest(DeclT* Latest) {
          // Convert A -> Latest -> B into A -> Latest
          getLink(Latest->getFirstDecl()).setLatest(Latest);
        }
        static void skipPrev(DeclT* Next) {
          // Convert A -> B -> Next into A -> Next
          DeclT* Skip = Next->getPreviousDecl();
          getLink(Next).setPrevious(Skip->getPreviousDecl());
        }
        static void setFirst(DeclT* First) {
          // Convert A -> First -> B into First -> B
          DeclT* Latest = First->getMostRecentDecl();
          getLink(First)
            = DeclLink(DeclLink::LatestLink, First->getASTContext());
          getLink(First).setLatest(Latest);
        }
      };

      assert(R != R->getFirstDecl() && "Cannot remove only redecl from chain");

      const bool isdef = isDefinition(R);

      // In the following cases, A marks the first, Z the most recent and
      // R the decl to be removed from the chain.
      DeclT* Prev = R->getPreviousDecl();
      if (R == R->getMostRecentDecl()) {
        // A -> .. -> R
        RedeclDerived::setLatest(Prev);
      } else {
        // Find the next redecl, starting at the end
        DeclT* Next = R->getMostRecentDecl();
        while (Next && Next->getPreviousDecl() != R)
          Next = Next->getPreviousDecl();
        if (!Next) {
          // R is not (yet?) wired up.
          return;
        }

        if (R->getPreviousDecl()) {
          // A -> .. -> R -> .. -> Z
          RedeclDerived::skipPrev(Next);
        } else {
          assert(R->getFirstDecl() == R && "Logic error");
          // R -> .. -> Z
          RedeclDerived::setFirst(Next);
        }
      }
      // If the decl was the definition, the other decl might have their
      // DefinitionData pointing to it.
      // This is really need only if DeclT is a TagDecl or derived.
      if (isdef) {
        resetDefinitionData(Prev);
      }
    }
    void removeRedeclFromChain(...) {
      llvm_unreachable("setLatestDeclImpl on non-redeclarable declaration");
    }

    ///\brief Removes given declaration from the chain of redeclarations.
    /// Rebuilds the chain and sets properly first and last redeclaration.
    /// @param[in] R - The redeclarable, its chain to be rebuilt.
    /// @param[in] DC - Remove the redecl's lookup entry from this DeclContext.
    ///
    ///\returns the most recent redeclaration in the new chain.
    ///
    template <typename T>
    bool VisitRedeclarable(clang::Redeclarable<T>* R, DeclContext* DC) {
      if (R->getFirstDecl() == R) {
        // This is the only element in the chain.
        return true;
      }

      T* MostRecent = R->getMostRecentDecl();
      T* MostRecentNotThis = MostRecent;
      if (MostRecentNotThis == R)
        MostRecentNotThis = R->getPreviousDecl();

      if (StoredDeclsMap* Map = DC->getPrimaryContext()->getLookupPtr()) {
        // Make sure we update the lookup maps, because the removed decl might
        // be registered in the lookup and again findable.
        NamedDecl* ND = (T*)R;
        DeclarationName Name = ND->getDeclName();
        if (!Name.isEmpty()) {
          StoredDeclsMap::iterator Pos = Map->find(Name);
          if (Pos != Map->end() && !Pos->second.isNull()) {
            DeclContext::lookup_result decls = Pos->second.getLookupResult();

            for (DeclContext::lookup_result::iterator I = decls.begin(),
                    E = decls.end(); I != E; ++I) {
              // FIXME: A decl meant to be added in the lookup already exists
              // in the lookup table. My assumption is that the DeclUnloader
              // adds it here. This needs to be investigated mode. For now
              // std::find gets promoted from assert to condition :)
              if (*I == ND && std::find(decls.begin(), decls.end(),
                                        MostRecentNotThis)
                  == decls.end()) {
                // The decl was registered in the lookup, update it.
                *I = MostRecentNotThis;
                break;
              }
            }
          }
        }
      }

      // Set a new latest redecl.
      removeRedeclFromChain((T*)R);
#ifndef NDEBUG
      // Validate redecl chain by iterating through it.
      std::set<clang::Redeclarable<T>*> CheckUnique;
      (void)CheckUnique;
      for (auto&& RD: MostRecentNotThis->redecls()) {
        assert(CheckUnique.insert(RD).second && "Dupe redecl chain element");
        (void)RD;
      }
#endif
      return true;
    }

    /// @}

  private:
    ///\brief Function that collects the files which we must reread from disk.
    ///
    /// For example: We must uncache the cached include, which brought a
    /// declaration or a macro diretive definition in the AST.
    ///\param[in] Loc - The source location of the unloaded declaration.
    ///
    void CollectFilesToUncache(SourceLocation Loc);
  };

  DeclUnloader::~DeclUnloader() {
    SourceManager& SM = m_Sema->getSourceManager();
    for (FileIDs::iterator I = m_FilesToUncache.begin(),
           E = m_FilesToUncache.end(); I != E; ++I) {
      // We need to reset the cache
      SM.invalidateCache(*I);
    }
  }

  void DeclUnloader::CollectFilesToUncache(SourceLocation Loc) {
    if (!m_CurTransaction)
      return;
    const SourceManager& SM = m_Sema->getSourceManager();
    FileID FID = SM.getFileID(SM.getSpellingLoc(Loc));
    if (!FID.isInvalid() && FID >= m_CurTransaction->getBufferFID()
        && !m_FilesToUncache.count(FID))
      m_FilesToUncache.insert(FID);
  }

  bool DeclUnloader::VisitDecl(Decl* D) {
    assert(D && "The Decl is null");
    CollectFilesToUncache(D->getLocStart());

    DeclContext* DC = D->getLexicalDeclContext();

    bool Successful = true;
    if (DC->containsDecl(D))
      DC->removeDecl(D);

    // With the bump allocator this is nop.
    if (Successful)
      m_Sema->getASTContext().Deallocate(D);
    return Successful;
  }

  bool DeclUnloader::VisitNamedDecl(NamedDecl* ND) {
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

      if (utils::Analyze::isOnScopeChains(ND, *m_Sema))
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
          for (StoredDeclsList::DeclsTy::const_iterator I = Vec->begin(),
                 E = Vec->end(); I != E; ++I)
            if (*I == ND)
              Pos->second.remove(ND);
        }
        if (Pos->second.isNull() ||
            (Pos->second.getAsVector() && !Pos->second.getAsVector()->size()))
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

  bool DeclUnloader::VisitDeclaratorDecl(DeclaratorDecl* DD) {
    // VisitDeclaratorDecl: ValueDecl
    auto found = std::find(m_Sema->UnusedFileScopedDecls.begin(/*ExtSource*/0,
                                                               /*Local*/true),
                           m_Sema->UnusedFileScopedDecls.end(), DD);
    if (found != m_Sema->UnusedFileScopedDecls.end())
      m_Sema->UnusedFileScopedDecls.erase(found,
                                          m_Sema->UnusedFileScopedDecls.end());

    return VisitValueDecl(DD);
  }

  bool DeclUnloader::VisitUsingShadowDecl(UsingShadowDecl* USD) {
    // UsingShadowDecl: NamedDecl, Redeclarable
    bool Successful = true;
    // FIXME: This is needed when we have newest clang:
    //Successful = VisitRedeclarable(USD, USD->getDeclContext());
    Successful &= VisitNamedDecl(USD);

    // Unregister from the using decl that it shadows.
    USD->getUsingDecl()->removeShadowDecl(USD);

    return Successful;
  }

  bool DeclUnloader::VisitTypedefNameDecl(TypedefNameDecl* TND) {
    // TypedefNameDecl: TypeDecl, Redeclarable
    bool Successful = VisitRedeclarable(TND, TND->getDeclContext());
    Successful &= VisitTypeDecl(TND);
    return Successful;
  }


  bool DeclUnloader::VisitVarDecl(VarDecl* VD) {
    // llvm::Module cannot contain:
    // * variables and parameters with dependent context;
    // * mangled names for parameters;
    if (!isa<ParmVarDecl>(VD) && !VD->getDeclContext()->isDependentContext()) {
      // Cleanup the module if the transaction was committed and code was
      // generated. This has to go first, because it may need the AST
      // information which we will remove soon. (Eg. mangleDeclName iterates the
      // redecls)
      GlobalDecl GD(VD);
      MaybeRemoveDeclFromModule(GD);
    }

    // VarDecl : DeclaratiorDecl, Redeclarable
    bool Successful = VisitRedeclarable(VD, VD->getDeclContext());
    Successful &= VisitDeclaratorDecl(VD);

    return Successful;
  }

  namespace {
    typedef llvm::SmallVector<VarDecl*, 2> Vars;
    class StaticVarCollector : public RecursiveASTVisitor<StaticVarCollector> {
      Vars& m_V;
    public:
      StaticVarCollector(FunctionDecl* FD, Vars& V) : m_V(V) {
        TraverseStmt(FD->getBody());
      }
      bool VisitDeclStmt(DeclStmt* DS) {
        for(DeclStmt::decl_iterator I = DS->decl_begin(), E = DS->decl_end();
            I != E; ++I)
          if (VarDecl* VD = dyn_cast<VarDecl>(*I))
            if (VD->isStaticLocal())
              m_V.push_back(VD);
        return true;
      }
    };
  }
  bool DeclUnloader::VisitFunctionDecl(FunctionDecl* FD) {
    // The Structors need to be handled differently.
    if (!isa<CXXConstructorDecl>(FD) && !isa<CXXDestructorDecl>(FD)) {
      // Cleanup the module if the transaction was committed and code was
      // generated. This has to go first, because it may need the AST info
      // which we will remove soon. (Eg. mangleDeclName iterates the redecls)
      GlobalDecl GD(FD);
      MaybeRemoveDeclFromModule(GD);
      // Handle static locals. void func() { static int var; } is represented in
      // the llvm::Module is a global named @func.var
      Vars V;
      StaticVarCollector c(FD, V);
      for (Vars::iterator I = V.begin(), E = V.end(); I != E; ++I) {
        GlobalDecl GD(*I);
        MaybeRemoveDeclFromModule(GD);
      }
    }
    // VisitRedeclarable() will mess around with this!
    bool wasCanonical = FD->isCanonicalDecl();
    // FunctionDecl : DeclaratiorDecl, DeclContext, Redeclarable
    // We start with the decl context first, because parameters are part of the
    // DeclContext and when trying to remove them we need the full redecl chain
    // still in place.
    bool Successful = VisitDeclContext(FD);
    Successful &= VisitRedeclarable(FD, FD->getDeclContext());
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
        assert(!self->findSpecialization(args->asArray(),  InsertPos)
               && "Finds the removed decl again!");
#endif
      }
    };

    if (FD->isFunctionTemplateSpecialization() && wasCanonical) {
      // Only the canonical declarations are registered in the list of the
      // specializations.
      FunctionTemplateDecl* FTD
        = FD->getTemplateSpecializationInfo()->getTemplate();
      // The canonical declaration of every specialization is registered with
      // the FunctionTemplateDecl.
      // Note this might unload too much in the case:
      //   template<typename T> T f(){ return T();}
      //   template<> int f();
      //   template<> int f() { return 0;}
      // when the template specialization was forward declared the canonical
      // becomes the first forward declaration. If the canonical forward
      // declaration was declared outside the set of the decls to unload we have
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

  bool DeclUnloader::VisitCXXConstructorDecl(CXXConstructorDecl* CXXCtor) {
    // Cleanup the module if the transaction was committed and code was
    // generated. This has to go first, because it may need the AST information
    // which we will remove soon. (Eg. mangleDeclName iterates the redecls)

    // Brute-force all possibly generated ctors.
    // Ctor_Complete            Complete object ctor.
    // Ctor_Base                Base object ctor.
    // Ctor_Comdat              The COMDAT used for ctors.
    GlobalDecl GD(CXXCtor, Ctor_Complete);
    MaybeRemoveDeclFromModule(GD);
    GD = GlobalDecl(CXXCtor, Ctor_Base);
    MaybeRemoveDeclFromModule(GD);
    GD = GlobalDecl(CXXCtor, Ctor_Comdat);
    MaybeRemoveDeclFromModule(GD);

    bool Successful = VisitCXXMethodDecl(CXXCtor);
    return Successful;
  }

  bool DeclUnloader::VisitCXXDestructorDecl(CXXDestructorDecl* CXXDtor) {
    // Cleanup the module if the transaction was committed and code was
    // generated. This has to go first, because it may need the AST information
    // which we will remove soon. (Eg. mangleDeclName iterates the redecls)

    // Brute-force all possibly generated dtors.
    // Dtor_Deleting            Deleting dtor.
    // Dtor_Complete            Complete object dtor.
    // Dtor_Base                Base object dtor.
    // Dtor_Comdat              The COMDAT used for dtors.
    GlobalDecl GD(CXXDtor, Dtor_Deleting);
    MaybeRemoveDeclFromModule(GD);
    GD = GlobalDecl(CXXDtor, Dtor_Complete);
    MaybeRemoveDeclFromModule(GD);
    GD = GlobalDecl(CXXDtor, Dtor_Base);
    MaybeRemoveDeclFromModule(GD);
    GD = GlobalDecl(CXXDtor, Dtor_Comdat);
    MaybeRemoveDeclFromModule(GD);

    bool Successful = VisitCXXMethodDecl(CXXDtor);
    return Successful;
  }

  bool DeclUnloader::VisitDeclContext(DeclContext* DC) {
    bool Successful = true;
    typedef llvm::SmallVector<Decl*, 64> Decls;
    Decls declsToErase;
    // Removing from single-linked list invalidates the iterators.
    for (DeclContext::decl_iterator I = DC->noload_decls_begin();
         I != DC->noload_decls_end(); ++I) {
      declsToErase.push_back(*I);
    }

    for(Decls::reverse_iterator I = declsToErase.rbegin(),
          E = declsToErase.rend(); I != E; ++I) {
      Successful = Visit(*I) && Successful;
      assert(Successful);
    }
    return Successful;
  }

  bool DeclUnloader::VisitNamespaceDecl(NamespaceDecl* NSD) {
    // NamespaceDecl: NamedDecl, DeclContext, Redeclarable
    bool Successful = VisitDeclContext(NSD);
    Successful &= VisitRedeclarable(NSD, NSD->getDeclContext());
    Successful &= VisitNamedDecl(NSD);

    return Successful;
  }

  bool DeclUnloader::VisitTagDecl(TagDecl* TD) {
    // TagDecl: TypeDecl, DeclContext, Redeclarable
    bool Successful = VisitDeclContext(TD);
    Successful &= VisitRedeclarable(TD, TD->getDeclContext());
    Successful &= VisitTypeDecl(TD);
    return Successful;
  }

  bool DeclUnloader::VisitRecordDecl(RecordDecl* RD) {
    if (RD->isInjectedClassName())
      return true;

    /// The injected class name in C++ is the name of the class that
    /// appears inside the class itself. For example:
    ///
    /// \code
    /// struct C {
    ///   // C is implicitly declared here as a synonym for the class name.
    /// };
    ///
    /// C::C c; // same as "C c;"
    /// \endcode
    // It is another question why it is on the redecl chain.
    // The test show it can be either:
    // ... <- InjectedC <- C <- ..., i.e previous decl or
    // ... <- C <- InjectedC <- ...
    RecordDecl* InjectedRD = RD->getPreviousDecl();
    if (!(InjectedRD && InjectedRD->isInjectedClassName())) {
      InjectedRD = RD->getMostRecentDecl();
      while (InjectedRD) {
        if (InjectedRD->isInjectedClassName()
            && InjectedRD->getPreviousDecl() == RD)
          break;
        InjectedRD = InjectedRD->getPreviousDecl();
      }
    }

    bool Successful = true;
    if (InjectedRD) {
      assert(InjectedRD->isInjectedClassName() && "Not injected classname?");
      Successful &= VisitRedeclarable(InjectedRD, InjectedRD->getDeclContext());
    }

    Successful &= VisitTagDecl(RD);
    return Successful;
  }

  void DeclUnloader::MaybeRemoveDeclFromModule(GlobalDecl& GD) const {
    if (!m_CurTransaction
        || !m_CurTransaction->getModule()) // syntax-only mode exit
      return;
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

      // Handle static locals. void func() { static int var; } is represented in
      // the llvm::Module is a global named @func.var
      if (const VarDecl* VD = dyn_cast<VarDecl>(GD.getDecl()))
        if (VD->isStaticLocal()) {
          std::string functionMangledName;
          GlobalDecl FDGD(cast<FunctionDecl>(VD->getDeclContext()));
          utils::Analyze::maybeMangleDeclName(FDGD, functionMangledName);
          mangledName = functionMangledName + "." + mangledName;
        }

      llvm::Module* M = m_CurTransaction->getModule();
      GlobalValue* GV = M->getNamedValue(mangledName);
      if (GV) { // May be deferred decl and thus 0
        GlobalValueEraser GVEraser(m_CodeGen);
        GVEraser.EraseGlobalValue(GV);
      }
    }
  }

  bool DeclUnloader::VisitMacro(Transaction::MacroDirectiveInfo MacroD) {
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

  bool DeclUnloader::VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl* R){
    // RedeclarableTemplateDecl: TemplateDecl, Redeclarable
    bool Successful = VisitRedeclarable(R, R->getDeclContext());
    Successful &= VisitTemplateDecl(R);
    return Successful;
  }

  bool DeclUnloader::VisitFunctionTemplateDecl(FunctionTemplateDecl* FTD) {
    bool Successful = true;

    // Remove specializations:
    for (FunctionTemplateDecl::spec_iterator I = FTD->spec_begin(),
           E = FTD->spec_end(); I != E; ++I)
      Successful &= Visit(*I);

    Successful &= VisitRedeclarableTemplateDecl(FTD);
    Successful &= VisitFunctionDecl(FTD->getTemplatedDecl());
    return Successful;
  }

  bool DeclUnloader::VisitClassTemplateDecl(ClassTemplateDecl* CTD) {
    // ClassTemplateDecl: TemplateDecl, Redeclarable
    bool Successful = true;
    // Remove specializations:
    for (ClassTemplateDecl::spec_iterator I = CTD->spec_begin(),
           E = CTD->spec_end(); I != E; ++I)
      Successful &= Visit(*I);

    Successful &= VisitRedeclarableTemplateDecl(CTD);
    Successful &= Visit(CTD->getTemplatedDecl());
    return Successful;
  }

  namespace {
  // A template specialization is attached to the list of specialization of
  // the templated class.
  //
  class ClassTemplateDeclExt : public ClassTemplateDecl {
  public:
    static void removeSpecialization(ClassTemplateDecl* self,
                                     ClassTemplateSpecializationDecl* spec) {
      assert(!isa<ClassTemplatePartialSpecializationDecl>(spec) &&
             "Use removePartialSpecialization");
      assert(self && spec && "Cannot be null!");
      assert(spec == spec->getCanonicalDecl()
             && "Not the canonical specialization!?");
      typedef llvm::SmallVector<ClassTemplateSpecializationDecl*, 4> Specializations;
      typedef llvm::FoldingSetVector<ClassTemplateSpecializationDecl> Set;

      ClassTemplateDeclExt* This = (ClassTemplateDeclExt*) self;
      Specializations specializations;
      Set& specs = This->getSpecializations();

      if (!specs.size()) // nothing to remove
        return;

      // Collect all the specializations without the one to remove.
      for(Set::iterator I = specs.begin(),E = specs.end(); I != E; ++I){
        if (&*I != spec)
          specializations.push_back(&*I);
      }

      This->getSpecializations().clear();

      //Readd the collected specializations.
      void* InsertPos = 0;
      ClassTemplateSpecializationDecl* CTSD = 0;
      for (size_t i = 0, e = specializations.size(); i < e; ++i) {
        CTSD = specializations[i];
        assert(CTSD && "Must not be null.");
        // Avoid assertion on add.
        CTSD->SetNextInBucket(0);
        This->AddSpecialization(CTSD, InsertPos);
      }
    }

    static void removePartialSpecialization(ClassTemplateDecl* self,
                                 ClassTemplatePartialSpecializationDecl* spec) {
      assert(self && spec && "Cannot be null!");
      assert(spec == spec->getCanonicalDecl()
             && "Not the canonical specialization!?");
      typedef llvm::SmallVector<ClassTemplatePartialSpecializationDecl*, 4>
        Specializations;
      typedef llvm::FoldingSetVector<ClassTemplatePartialSpecializationDecl> Set;

      ClassTemplateDeclExt* This = (ClassTemplateDeclExt*) self;
      Specializations specializations;
      Set& specs = This->getPartialSpecializations();

      if (!specs.size()) // nothing to remove
        return;

      // Collect all the specializations without the one to remove.
      for(Set::iterator I = specs.begin(),E = specs.end(); I != E; ++I){
        if (&*I != spec)
          specializations.push_back(&*I);
      }

      This->getPartialSpecializations().clear();

      //Readd the collected specializations.
      void* InsertPos = 0;
      ClassTemplatePartialSpecializationDecl* CTPSD = 0;
      for (size_t i = 0, e = specializations.size(); i < e; ++i) {
        CTPSD = specializations[i];
        assert(CTPSD && "Must not be null.");
        // Avoid assertion on add.
        CTPSD->SetNextInBucket(0);
        This->AddPartialSpecialization(CTPSD, InsertPos);
      }
    }
  };
  } // end anonymous namespace


  bool DeclUnloader::VisitClassTemplateSpecializationDecl(
                                        ClassTemplateSpecializationDecl* CTSD) {
    // ClassTemplateSpecializationDecl: CXXRecordDecl, FoldingSet
    bool Successful = VisitCXXRecordDecl(CTSD);
    ClassTemplateSpecializationDecl* CanonCTSD =
      static_cast<ClassTemplateSpecializationDecl*>(CTSD->getCanonicalDecl());
    if (auto D = dyn_cast<ClassTemplatePartialSpecializationDecl>(CanonCTSD))
      ClassTemplateDeclExt::removePartialSpecialization(
                                                    D->getSpecializedTemplate(),
                                                    D);
    else
      ClassTemplateDeclExt::removeSpecialization(CTSD->getSpecializedTemplate(),
                                                 CanonCTSD);
    return Successful;
  }
} // end namespace clang

namespace cling {
  TransactionUnloader::TransactionUnloader(Sema* S, clang::CodeGenerator* CG)
    : m_Sema(S), m_CodeGen(CG) {
  }

  TransactionUnloader::~TransactionUnloader() {
  }

  bool TransactionUnloader::RevertTransaction(Transaction* T) {
    if (Transaction* Parent = T->getParent()) {
      Parent->removeNestedTransaction(T);
      T->setParent(0);
    }

    DeclUnloader DeclU(m_Sema, m_CodeGen, T);
    bool Successful = true;

    for (Transaction::const_reverse_iterator I = T->rdecls_begin(),
           E = T->rdecls_end(); I != E; ++I) {
      const Transaction::ConsumerCallInfo& Call = I->m_Call;
      const DeclGroupRef& DGR = (*I).m_DGR;

      if (Call == Transaction::kCCIHandleVTable)
        continue;
      // The non templated classes come through HandleTopLevelDecl and
      // HandleTagDeclDefinition, this is why we need to filter.
      if (Call == Transaction::kCCIHandleTagDeclDefinition)
        if (const CXXRecordDecl* D
            = dyn_cast<CXXRecordDecl>(DGR.getSingleDecl()))
          if (D->getTemplateSpecializationKind() == TSK_Undeclared)
            continue;

      if (Call == Transaction::kCCINone)
        RevertTransaction(*T->rnested_begin());

      for (DeclGroupRef::const_iterator
             Di = DGR.end() - 1, E = DGR.begin() - 1; Di != E; --Di) {
        // Get rid of the declaration. If the declaration has name we should
        // heal the lookup tables as well
        Successful = DeclU.UnloadDecl(*Di) && Successful;
#ifndef NDEBUG
        assert(Successful && "Cannot handle that yet!");
#endif
      }
    }
    assert(T->rnested_begin() == T->rnested_end() && "nested transactions mismatch");

    for (Transaction::const_reverse_macros_iterator MI = T->rmacros_begin(),
           ME = T->rmacros_end(); MI != ME; ++MI) {
      // Get rid of the macro definition
      Successful = DeclU.UnloadMacro(*MI) && Successful;
#ifndef NDEBUG
      assert(Successful && "Cannot handle that yet!");
#endif
    }

#ifndef NDEBUG
    //FIXME: Move the nested transaction marker out of the decl lists and
    // reenable this assertion.
    //size_t DeclSize = std::distance(T->decls_begin(), T->decls_end());
    //if (T->getCompilationOpts().CodeGenerationForModule)
    //  assert (!DeclSize && "No parsed decls must happen in parse for module");
#endif

    //FIXME: Terrible hack, we *must* get rid of parseForModule by implementing
    // a header file generator in cling.
    for (Transaction::const_reverse_iterator I = T->deserialized_rdecls_begin(),
           E = T->deserialized_rdecls_end(); I != E; ++I) {
      const DeclGroupRef& DGR = (*I).m_DGR;
      for (DeclGroupRef::const_iterator
             Di = DGR.end() - 1, E = DGR.begin() - 1; Di != E; --Di) {
        // We only want to revert all that came through parseForModule, and
        // not the PCH.
        if (!(*Di)->isFromASTFile())
          Successful = DeclU.UnloadDecl(*Di) && Successful;
#ifndef NDEBUG
        assert(Successful && "Cannot handle that yet!");
#endif
      }
    }

    // Clean up the pending instantiations
    m_Sema->PendingInstantiations.clear();
    m_Sema->PendingLocalImplicitInstantiations.clear();

    // Cleanup the module from unused global values.
    // if (T->getModule()) {
    //   llvm::ModulePass* globalDCE = llvm::createGlobalDCEPass();
    //   globalDCE->runOnModule(*T->getModule());
    // }
    if (Successful)
      T->setState(Transaction::kRolledBack);
    else
      T->setState(Transaction::kRolledBackWithErrors);

    return Successful;
  }

  bool TransactionUnloader::UnloadDecl(Decl* D) {
    DeclUnloader DeclU(m_Sema, m_CodeGen, 0);
    return DeclU.UnloadDecl(D);
  }
} // end namespace cling

