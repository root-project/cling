//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "DeclUnloader.h"

#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

#include "llvm/IR/Constants.h"

namespace cling {
using namespace clang;

bool DeclUnloader::UnloadDecl(Decl* D) {
  DiagnosticErrorTrap Trap(m_Sema->getDiagnostics());
  const bool Successful = Visit(D);
  if (Trap.hasErrorOccurred())
    m_Sema->getDiagnostics().Reset(true);
  return Successful;
}

namespace {
  static bool isDefinition(void*) {
    return false;
  }
  static bool isDefinition(clang::TagDecl* R) {
    return R->isCompleteDefinition();
  }
  // Flags for DeclUnloader::VisitNamedDecl
  enum {
    kVisitingSpecialization  = 1,  // Currently visiting specializations
    kLeaveScopeMap  = 2,  // Don't remove Decl from StoredDeclsMap of scope.
    kSkipNamespaceRemoval = 4, // Don't remove Decl from enclosing namespace(s)
    // Anything added here must increase bits in DeclUnloader::m_Flags
  };
}

class DeclUnloader::VisitorState {
  DeclUnloader& m_DU; // parent
  const unsigned m_F; // prior flags
public:
  VisitorState(DeclUnloader &DU, unsigned F) :  m_DU(DU), m_F(DU.m_Flags) {
    m_DU.m_Flags |= F;  // Or in the new flags
  }
  ~VisitorState() {
    m_DU.m_Flags = m_F; // Reset it to the way it was
  }
};
  
template <class DeclT>
void DeclUnloader::resetDefinitionData(DeclT*) {
}

template <>
void DeclUnloader::resetDefinitionData(clang::TagDecl* D) {
  if (clang::CXXRecordDecl* C = dyn_cast<CXXRecordDecl>(D)) {
    // It was allocated this way...
    ::operator delete(C->DefinitionData, D->getASTContext(),
                      sizeof(CXXRecordDecl::DefinitionData));
    for (C = C->getCanonicalDecl()->getMostRecentDecl(); C;
         C = C->getPreviousDecl()) {
      C->DefinitionData = nullptr;
      C->IsCompleteDefinition = 0;
    }
    assert(cast<CXXRecordDecl>(D)->DefinitionData == nullptr && "Not cleared");
  } else {
    for (clang::TagDecl* T = D->getCanonicalDecl()->getMostRecentDecl(); T;
         T = T->getPreviousDecl()) {
      T->IsCompleteDefinition = 0;
    }
  }
  assert(D->IsCompleteDefinition == 0 && "Not reset");
}

// RedeclLink is a protected member, which we need access to in
// removeRedeclFromChain (and VisitRedeclarable for checking in debug mode)
template<typename DeclT>
struct RedeclDerived : public Redeclarable<DeclT> {
  typedef typename Redeclarable<DeclT>::DeclLink DeclLink_t;
  static DeclLink_t& getLink(DeclT* LR) {
    Redeclarable<DeclT>* D = LR;
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
      = DeclLink_t(DeclLink_t::LatestLink, First->getASTContext());
    getLink(First).setLatest(Latest);
  }
  static DeclT *getNextRedeclaration(DeclT* R) {
    return getLink(R).getNext(R);
  }
};

// Copied and adapted from: ASTReaderDecl.cpp
template<typename DeclT>
void DeclUnloader::removeRedeclFromChain(DeclT* R) {
  assert(R != R->getFirstDecl() && "Cannot remove only redecl from chain");

  const bool isdef = isDefinition(R);

  // In the following cases, A marks the first, Z the most recent and
  // R the decl to be removed from the chain.
  DeclT* Prev = R->getPreviousDecl();
  if (R == R->getMostRecentDecl()) {
    // A -> .. -> R
    RedeclDerived<DeclT>::setLatest(Prev);
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
      RedeclDerived<DeclT>::skipPrev(Next);
    } else {
      assert(R->getFirstDecl() == R && "Logic error");
      // R -> .. -> Z
      RedeclDerived<DeclT>::setFirst(Next);
    }
  }
  // If the decl was the definition, the other decl might have their
  // DefinitionData pointing to it.
  // This is really need only if DeclT is a TagDecl or derived.
  if (isdef) {
    resetDefinitionData(Prev);
  }
}

///\brief Adds the previous declaration into the lookup map on DC.
/// @param[in] D - The decl that is being removed.
/// @param[in] DC - The DeclContext to add the previous declaration of D.
///\returns the previous declaration.
///
static Decl* handleRedelaration(Decl* D, DeclContext* DC) {
  NamedDecl* ND = dyn_cast<NamedDecl>(D);
  if (!ND)
    return nullptr;

  DeclarationName Name = ND->getDeclName();
  if (Name.isEmpty())
    return nullptr;

  NamedDecl* MostRecent = ND->getMostRecentDecl();
  NamedDecl* MostRecentNotThis = MostRecent;
  if (MostRecentNotThis == ND) {
    MostRecentNotThis = dyn_cast_or_null<NamedDecl>(ND->getPreviousDecl());
    if (!MostRecentNotThis || MostRecentNotThis == ND)
      return MostRecentNotThis;
  }

  // Mirror what DeclContext::removeDecl does, otherwise we'll get out
  // of synch for the call to removeDecl

  do {
    if (StoredDeclsMap* Map = DC->getPrimaryContext()->getLookupPtr()) {
      StoredDeclsMap::iterator Pos = Map->find(Name);
      if (Pos != Map->end() && !Pos->second.isNull()) {
        DeclContext::lookup_result decls = Pos->second.getLookupResult();
        // FIXME: A decl meant to be added in the lookup already exists
        // in the lookup table. My assumption is that the DeclUnloader
        // adds it here. This needs to be investigated mode. For now
        // std::find gets promoted from assert to condition :)
        // DeclContext::lookup_result::iterator is not an InputIterator
        // (const member, thus no op=(const iterator&)), thus we cannot use
        // std::find. MSVC actually cares!
        auto hasDecl = [](const DeclContext::lookup_result& Result,
                          const NamedDecl* Needle) -> bool {
          for (auto IDecl: Result) {
            if (IDecl == Needle)
              return true;
          }
          return false;
        };
        if (!hasDecl(decls, MostRecentNotThis) && hasDecl(decls, ND)) {
          // The decl was registered in the lookup, update it.
          Pos->second.HandleRedeclaration(MostRecentNotThis,
                                          /*IsKnownNewer*/ true);
        }
      }
    }
  } while (DC->isTransparentContext() && (DC = DC->getParent()));
  return MostRecentNotThis;
}

///\brief Removes given declaration from the chain of redeclarations.
/// Rebuilds the chain and sets properly first and last redeclaration.
/// @param[in] R - The redeclarable, its chain to be rebuilt.
/// @param[in] DC - Remove the redecl's lookup entry from this DeclContext.
///
///\returns the most recent redeclaration in the new chain.
///
template <typename T>
bool DeclUnloader::VisitRedeclarable(clang::Redeclarable<T>* R, DeclContext* DC) {
  if (R->getFirstDecl() == R) {
    // This is the only element in the chain.
    if (isDefinition((T*)R))
      resetDefinitionData((T*)R);
    return true;
  }

  // Make sure we update the lookup maps, because the removed decl might
  // be registered in the lookup and still findable.
  T* MostRecentNotThis = (T*)handleRedelaration((T*)R, DC);

  // Set a new latest redecl.
  removeRedeclFromChain((T*)R);

#ifndef NDEBUG
  // Validate redecl chain by iterating through it.
  // If we get here and T == clang::UsingShadowDecl
  // for (auto RD: MostRecentNotThis->redecls()) won't work
  // as the redecl_iterator has an ambigous isFirstDecl call
  // So we had to pull the guts out of redecl_iterator into here

  std::set<clang::Redeclarable<T>*> Unique;
  T* Current = MostRecentNotThis;
  T* Starter = Current;

  while (Current) {
    assert(Unique.insert(Current).second && "Dupe redecl chain element");
    Current = RedeclDerived<T>::getNextRedeclaration(Current);
    if (Current==Starter)
      Current = nullptr;
  }
#else
  (void)MostRecentNotThis; // templated function issues a lot -Wunused-variable
#endif

  return true;
}

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
        if (GlobalVariable* GVar = dyn_cast<GlobalVariable>(*I)) {
          GVar->setInitializer(0);
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
        // GA->getAliasee() is sometimes returning NULL on Windows
        if (llvm::Constant* C = GA->getAliasee())
          MarkConstant(C);
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

  DeclUnloader::~DeclUnloader() {
    SourceManager& SM = m_Sema->getSourceManager();
    for (FileIDs::iterator I = m_FilesToUncache.begin(),
           E = m_FilesToUncache.end(); I != E; ++I) {
      // We need to reset the cache
      SM.invalidateCache(*I);
    }
  }

  void DeclUnloader::CollectFilesToUncache(SourceLocation Loc) {
    if (!m_CurTransaction || Loc.isInvalid())
      return;
    const SourceManager& SM = m_Sema->getSourceManager();
    FileID FID = SM.getFileID(SM.getSpellingLoc(Loc));
    if (!FID.isInvalid() && FID >= m_CurTransaction->getBufferFID())
      m_FilesToUncache.insert(FID);
  }

  static void reportContext(const DeclContext* DC, llvm::raw_ostream& Out) {
    Out << DC->getDeclKindName();
    if (const NamedDecl* Ctx = dyn_cast<NamedDecl>(DC))
      Out << " '" << Ctx->getNameAsString() << "'";
    else
      Out << " (" << DC << ")";
  }

  static void reportErrors(Sema* Sema, Decl* D, const SourceLocation& Loc,
                           const llvm::SmallVector<DeclContext*, 4>& errors) {
    std::string errStr("success trying to remove ");
    llvm::raw_string_ostream Out(errStr);

    // We know its a NamedDecl as that's the only type that can set an error
    Out << D->getDeclKindName() << " '" << cast<NamedDecl>(D)->getNameAsString()
        << "' from ";

    if (errors.size() > 1) {
      Out << "{";
      llvm::SmallVector<DeclContext *, 4>::const_iterator itr = errors.begin(),
                                                          end = errors.end();
      while (true) {
        Out << " ";
        reportContext(*itr, Out);
        if (++itr == end)
          break;
        Out << ",";
      }
      Out << " }";
    } else
      reportContext(errors[0], Out);

    Sema->Diags.Report(Loc, diag::err_expected) << Out.str();
  }

  static SourceLocation getDeclLocation(Decl* D) {
    switch (D->getKind()) {
      case Decl::ClassTemplateSpecialization:
      case Decl::ClassTemplatePartialSpecialization: {
        auto* CTS = cast<ClassTemplateSpecializationDecl>(D);
        const SourceLocation Loc = CTS->getPointOfInstantiation();
        if (Loc.isValid())
          return Loc;
        if (CTS->getSpecializationKind() == clang::TSK_Undeclared)
          return SourceLocation();
        break;
      }
      case Decl::Function: {
          FunctionDecl* FD = cast<FunctionDecl>(D);
          if (FunctionTemplateSpecializationInfo *Info =
                                          FD->getTemplateSpecializationInfo()) {
            const SourceLocation Loc = Info->getPointOfInstantiation();
            if (Loc.isValid())
              return Loc;
          }
        break;
      }
      default:
        break;
    }
    return D->getLocStart();
  }

  bool DeclUnloader::VisitDecl(Decl* D) {
    assert(D && "The Decl is null");
    const SourceLocation Loc = getDeclLocation(D);
    CollectFilesToUncache(Loc);

    DeclContext* DC = D->getLexicalDeclContext();

    bool Successful = true;
    if (DC->containsDecl(D)) {
      llvm::SmallVector<DeclContext*, 4> errors;
      DC->removeDecl(D, &errors);
      if (!errors.empty())
        reportErrors(m_Sema, D, Loc, errors);
    }

    // With the bump allocator this is nop.
    if (Successful)
      m_Sema->getASTContext().Deallocate(D);
    return Successful;
  }

  // Remove a decl and possibly it's parent entry in lookup tables.
  static void eraseDeclFromMap(StoredDeclsMap* Map, NamedDecl* ND) {
    assert(Map && ND && "eraseDeclFromMap recieved NULL value(s)");
    // Make sure we the decl doesn't exist in the lookup tables.
    StoredDeclsMap::iterator Pos = Map->find(ND->getDeclName());
    if (Pos != Map->end()) {
      // Most decls only have one entry in their list, special case it.
      if (Pos->second.getAsDecl() == ND) {
        // This is the only decl, no need to call Pos->second.remove(ND);
        // as it only sets data to nullptr, just remove the entire entry
        Map->erase(Pos);
      }
      else if (StoredDeclsList::DeclsTy* Vec = Pos->second.getAsVector()) {
        // Otherwise iterate over the list with entries with the same name.
        for (NamedDecl* NDi : *Vec) {
          if (NDi == ND)
            Pos->second.remove(ND);
        }
        if (Vec->empty())
          Map->erase(Pos);
      }
      else if (Pos->second.isNull()) // least common case
        Map->erase(Pos);
    }
  }

#ifndef NDEBUG
  // Make sure we the decl doesn't exist in the lookup tables.
  static void checkDeclIsGone(StoredDeclsMap* Map, NamedDecl* ND) {
    assert(Map && ND && "checkDeclIsGone recieved NULL value(s)");
    StoredDeclsMap::iterator Pos = Map->find(ND->getDeclName());
    if ( Pos != Map->end()) {
      // Most decls only have one entry in their list, special case it.
      if (NamedDecl* OldD = Pos->second.getAsDecl())
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

  DeclContext* DeclUnloader::removeFromScope(NamedDecl* ND, bool Force) {
    DeclContext* DC = ND->getDeclContext();
    while (DC->isTransparentContext())
      DC = DC->getLookupParent();

    if (Force || ND->getIdentifier()) {
      if (Scope* S = m_Sema->getScopeForContext(DC))
        S->RemoveDecl(ND);

      if (utils::Analyze::isOnScopeChains(ND, *m_Sema))
        m_Sema->IdResolver.RemoveDecl(ND);
    }
    return DC;
  }

  bool DeclUnloader::VisitNamedDecl(NamedDecl* ND) {
    if (!VisitDecl(ND))
      return false;

    llvm::SetVector<StoredDeclsMap*> Maps;
    DeclContext* DC = removeFromScope(ND);
    if (LLVM_UNLIKELY(!(m_Flags & kLeaveScopeMap))) {
      if (StoredDeclsMap* Map = DC->getPrimaryContext()->getLookupPtr())
        Maps.insert(Map);
    }

    // Remove from all namespace's lookup maps going upwards.
    if (LLVM_UNLIKELY(!(m_Flags & kSkipNamespaceRemoval))) {
      NamespaceDecl* NS = dyn_cast<NamespaceDecl>(DC);
      while(DC && !NS) {
        DC = DC->getParent();
        NS = dyn_cast_or_null<NamespaceDecl>(DC);
      }

      if (NS && NS->isInline()) {
        // VisitDecl will have already done a removal on Lexical, so skip that
        DeclContext* Lexical = ND->getLexicalDeclContext();
        do {
          assert((NS->getFirstDecl() != NS ? !NS->getLookupPtr() : 1)
                 && "Has unique lookup ptr!");
          if (DC != Lexical) {
            // There is a chance that lookup ptr will not exist when rolling back
            // a bad Transaction. The question is whether we can stop the loop too?
            if (StoredDeclsMap* Map = NS->getFirstDecl()->getLookupPtr())
              Maps.insert(Map);
          }
          DC = DC->getParent();
          NS = dyn_cast<NamespaceDecl>(DC);
        } while (NS);
      }
    }

    // If the decl was removed make sure that we fix the lookup
    // DeclContexts like EnumDecls don't have lookup maps.
    for (StoredDeclsMap* Map : Maps) {
      eraseDeclFromMap(Map, ND);
#ifndef NDEBUG
      checkDeclIsGone(Map, ND);
#endif
    }

    return true;
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

    assert(USD->getTargetDecl() && "No target for UsingShadow");
    VisitorState VS(*this, (USD->getFirstDecl() == USD ? kLeaveScopeMap : 0) |
                    (wasInstatiatedBefore(getDeclLocation(USD->getTargetDecl()))
                    ? kSkipNamespaceRemoval : 0));

    bool Successful = VisitRedeclarable(USD, USD->getDeclContext());
    Successful &= VisitNamedDecl(USD);

    // Unregister from the using decl that it shadows.
    USD->getUsingDecl()->removeShadowDecl(USD);

    return Successful;
  }

  bool DeclUnloader::VisitUsingDecl(UsingDecl* UD) {
    // UsingDecl: NamedDecl, Mergeable<UsingDecl>
    bool Success = true;

    // UsingDecls currently are not Redeclarables, instead they overwrite the
    // entry in the shared lookup map.  The prior declarations still exist in
    // the previous context's decl list, so grab the prior value, and add
    // the entry in the lookup map as a final step.
    //
    // Save the previous Namespace and UsingDecl now.
    NamespaceDecl *PrvNS = nullptr;
    UsingDecl *PrvUD = nullptr;
    if (NamespaceDecl *NS =
                        dyn_cast_or_null<NamespaceDecl>(UD->getDeclContext())) {
      if ((PrvNS = NS->getPreviousDecl())) {
        const IdentifierInfo *ID = UD->getIdentifier();
        for (Decl* D : PrvNS->noload_decls()) {
          if (UsingDecl *Ud = dyn_cast<UsingDecl>(D)) {
            if (Ud->getIdentifier() == ID) {
              PrvUD = Ud;
              break;
            }
          }
        }
      }
    }

    llvm::SmallVector<UsingShadowDecl*, 12> Shadows(UD->shadow_begin(),
                                                    UD->shadow_end());
    for (UsingShadowDecl *USD : Shadows) {
      Success &= VisitUsingShadowDecl(USD);
      assert(Success);
    }

    Success &= VisitNamedDecl(UD);

    // Replace or re-add the prior UsingDecl entry.
    if (PrvNS && PrvUD) {
      // Only need to overwrite PrvNS as they all share a map
      // do {
        if (DeclContext *DC = PrvNS->getPrimaryContext()) {
          if (StoredDeclsMap *Map = DC->getLookupPtr()) {
            auto &DeclList = (*Map)[PrvUD->getDeclName()];
            if (!DeclList.HandleRedeclaration(PrvUD, true))
              DeclList.AddSubsequentDecl(PrvUD);
          }
        }
      //  PrvNS = PrvNS->getPreviousDecl();
      // } while (PrvNS);
    }

    return Success;
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
    const bool DepContext = VD->getDeclContext()->isDependentContext();
    if (!isa<ParmVarDecl>(VD) && !DepContext) {
      // Cleanup the module if the transaction was committed and code was
      // generated. This has to go first, because it may need the AST
      // information which we will remove soon. (Eg. mangleDeclName iterates the
      // redecls)
      GlobalDecl GD(VD);
      MaybeRemoveDeclFromModule(GD);
    }

    // VarDecl : DeclaratiorDecl, Redeclarable
    bool Successful = VisitRedeclarable(VD, VD->getDeclContext());
    if (!DepContext)
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

    static const TagType* getTagPointedToo(const Type* T) {
      if ((T = T->getPointeeType().getTypePtrOrNull())) {
        const Type* Resolved;
        do {
          Resolved = T;
          T = T->getPointeeType().getTypePtrOrNull();
        } while (T);
        return Resolved->getAs<TagType>();
      }
      return nullptr;
    }
  }

  bool DeclUnloader::VisitReturnValue(const QualType QT, Decl* Parent) {
    // struct FirstDecl* function();
    // In the example above, FirstDecl* will be declared as the function is
    // parsed, and if FirstDecl* never gets a definiton, it's declaration
    // will never be removed, so we have to check for that case.
    const Type* T = QT.getTypePtrOrNull();
    if (T && !T->isVoidType()) {
      if (const TagType* TT = getTagPointedToo(T)) {
        TagDecl* Tag = TT->getDecl();
        // So we have a TagDecl, was it embedded, and is it still undefined?
        if (Tag->isEmbeddedInDeclarator() && !Tag->isCompleteDefinition()) {
          // Possibly overkill, but just to be safe
          if (Tag->getFirstDecl() == Tag) {
            // Only unload it the first time it appeared with a function
            const SourceManager& SM = m_Sema->getSourceManager();
            if (!SM.isBeforeInTranslationUnit(Tag->getLocStart(),
                                              Parent->getLocStart())) {
              return Visit(Tag);
            }
          }
        }
      }
    }
    return true;
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

    Successful &= VisitReturnValue(FD->getReturnType(), FD);

    // Remove new and delete (all variants). These are 'anonymous' and not
    // removed in VisitNamedDecl. Built-in operator new & delete are still
    // present in m_Sema, so don't touch m_Sema->GlobalNewDeleteDeclared
    if (!FD->getIdentifier()) {
      const DeclarationName Name = FD->getDeclName();
      DeclarationNameTable& Table = m_Sema->getASTContext().DeclarationNames;
      if (Name == Table.getCXXOperatorName(OO_New) ||
          Name == Table.getCXXOperatorName(OO_Array_New) ||
          Name == Table.getCXXOperatorName(OO_Delete) ||
          Name == Table.getCXXOperatorName(OO_Array_Delete)) {
        (void)removeFromScope(FD, true);
      }
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

    // Removing from single-linked list invalidates the iterators.
    typedef llvm::SmallVector<Decl*, 64> Decls;
    Decls declsToErase(DC->noload_decls_begin(), DC->noload_decls_end());

    for(Decls::reverse_iterator I = declsToErase.rbegin(),
          E = declsToErase.rend(); I != E; ++I) {
      Successful = Visit(*I) && Successful;
      assert(Successful);
    }
    return Successful;
  }

  bool DeclUnloader::wasInstatiatedBefore(SourceLocation Loc) const {
    if (m_CurTransaction && Loc.isValid()) {
      SourceManager &SManger = m_Sema->getSourceManager();
      SourceLocation TLoc = m_CurTransaction->getSourceStart(SManger);
      if (TLoc.isValid())
        return SManger.isBeforeInTranslationUnit(Loc, TLoc);
    }
    return false;
  }

  bool DeclUnloader::VisitNamespaceDecl(NamespaceDecl* NSD) {
    // NamespaceDecl: NamedDecl, DeclContext, Redeclarable

    bool Successful = VisitRedeclarable(NSD, NSD->getDeclContext());
    Successful &= VisitDeclContext(NSD);
    Successful &= VisitNamedDecl(NSD);

    // Get these out of the caches
    if (NSD == m_Sema->getStdNamespace())
      m_Sema->StdNamespace = NSD->getPreviousDecl();
    m_Sema->KnownNamespaces.erase(NSD);

    return Successful;
  }

  bool DeclUnloader::VisitLinkageSpecDecl(LinkageSpecDecl* LSD) {
    // LinkageSpecDecl: DeclContext

    // Re-add any previous declarations so they are reachable throughout the
    // translation unit. Also remove any global variables from:
    // m_Sema->Context.getExternCContextDecl()

    if (LSD->isExternCContext()) {
      // Sadly ASTContext::getExternCContextDecl will create if it doesn't exist
      // Hopefully LSD->isExternCContext() means that it already does exist
      ExternCContextDecl* ECD = m_Sema->Context.getExternCContextDecl();
      StoredDeclsMap* Map = ECD ? ECD->getLookupPtr() : nullptr;
      
      for (Decl* D : LSD->noload_decls()) {
        if (NamedDecl* ND = dyn_cast<NamedDecl>(D)) {

          // extern "C" linkage goes in the translation unit
          DeclContext* DC = m_Sema->getASTContext().getTranslationUnitDecl();
          handleRedelaration(ND, DC);
          if (Map)
            eraseDeclFromMap(Map, ND);
        }
      }
    }

    bool Successful = VisitDeclContext(LSD);
    Successful &= VisitDecl(LSD);
    return Successful;
  }

  bool DeclUnloader::VisitFriendDecl(FriendDecl* FD) {
    // FriendDecl: Decl
    
    // Remove the friend declarations
    bool Successful = true;

    if (TypeSourceInfo* TI = FD->getFriendType()) {
      if (const Type* T = TI->getType().getTypePtrOrNull()) {
        if (const TagType* RT = T->getAs<TagType>()) {
          TagDecl *F = RT->getDecl();
          // If the friend is a class and embedded in the parent and not defined
          // then there is no further declaration so it must be unloaded now.
          if (F->isEmbeddedInDeclarator() &&  !F->isCompleteDefinition()) {
            // Avoid recursion: class A { class B { friend class A; } }
            TagDecl* Parent = dyn_cast_or_null<TagDecl>(FD->getDeclContext());
            if (!Parent || F != Parent->getDeclContext())
              Successful &= Visit(F);
          }
        }
      }
    } else if (NamedDecl* ND = FD->getFriendDecl())
      Successful &= Visit(ND);

    // Is it possible to unload a friend but not the parent class?
    // This requires adding DeclUnloader as a friend to FriendDecl
#if 0
    // Unlink the FriendDecl from linked list in CXXRecordDecl
    TagDecl* Parent = dyn_cast_or_null<TagDecl>(FD->getDeclContext());
    if (CXXRecordDecl* CD = dyn_cast_or_null<CXXRecordDecl>(Parent)) {
      Successful = false;
      if (FriendDecl *First = CD->getFirstFriend()) {
        if (First != FD) {
          FriendDecl *Prev = First;
          for (FriendDecl* Cur : CD->friends()) {
            if (Cur == FD) {
              Prev->NextFriend = Cur->NextFriend;
              Successful = true;
              break;
            }
            Prev = Cur;
          }
        } else {
          Successful = true;
          CD->data().FirstFriend = nullptr;
        }
      }

      // Arrived back here through recursion?
      if (!Successful)
        return true;

      Successful = true;
    }
#endif

    Successful &= VisitDecl(FD);
    return Successful;
  }

  bool DeclUnloader::VisitTagDecl(TagDecl* TD) {
    // TagDecl: TypeDecl, DeclContext, Redeclarable
    bool Successful = VisitRedeclarable(TD, TD->getDeclContext());
    Successful &= VisitDeclContext(TD);
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

  bool DeclUnloader::VisitCXXRecordDecl(CXXRecordDecl* RD) {
    // CXXRecordDecl: RecordDecl

    // clang caches some decls, so we have to remove them there as well
    if (RD == m_Sema->CXXTypeInfoDecl)
      m_Sema->CXXTypeInfoDecl = m_Sema->CXXTypeInfoDecl->getPreviousDecl();
    else if (RD == m_Sema->MSVCGuidDecl)
      m_Sema->MSVCGuidDecl = m_Sema->MSVCGuidDecl->getPreviousDecl();
    else if (RD == m_Sema->getStdBadAlloc())
      m_Sema->StdBadAlloc = m_Sema->getStdBadAlloc()->getPreviousDecl();

    return VisitRecordDecl(RD);
  }

  void DeclUnloader::MaybeRemoveDeclFromModule(GlobalDecl& GD) const {
    if (!m_CurTransaction
        || !m_CurTransaction->getModule()) // syntax-only mode exit
      return;

    using namespace llvm;

    if (const auto VD = dyn_cast_or_null<clang::ValueDecl>(GD.getDecl())) {
      const QualType QT = VD->getType();
      if (!QT.isNull() && QT->isDependentType()) {
        // The module cannot contain symbols for dependent decls.
        return;
      }
    }

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

      // Handle static locals. void func() { static int var; } is represented
      // in the llvm::Module is a global named @func.var
      if (const VarDecl* VD = dyn_cast<VarDecl>(GD.getDecl())) {
        if (VD->isStaticLocal()) {
          std::string functionMangledName;
          GlobalDecl FDGD(cast<FunctionDecl>(VD->getDeclContext()));
          utils::Analyze::maybeMangleDeclName(FDGD, functionMangledName);
          mangledName = functionMangledName + "." + mangledName;
        }
      }

      llvm::Module* M = m_CurTransaction->getModule();
      GlobalValue* GV = M->getNamedValue(mangledName);
      if (GV) { // May be deferred decl and thus 0
        GlobalValueEraser GVEraser(m_CodeGen);
        GVEraser.EraseGlobalValue(GV);
      }
      m_CodeGen->forgetDecl(GD);
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

  namespace {

  } // end anonymous namespace

  template <class DeclT>
  bool DeclUnloader::VisitSpecializations(DeclT *D) {
    llvm::SmallVector<typename DeclT::spec_iterator::pointer, 8> specs;
    for (typename DeclT::spec_iterator I = D->spec_begin(),
           E = D->spec_end(); I != E; ++I) {
        specs.push_back(*I);
    }

    bool Successful = true;
    VisitorState VS(*this, kVisitingSpecialization);
    for (typename DeclT::spec_iterator::pointer spec : specs)
      Successful &= Visit(spec);

    return Successful;
  }

  bool DeclUnloader::VisitFunctionTemplateDecl(FunctionTemplateDecl* FTD) {
    // FunctionTemplateDecl: TemplateDecl, Redeclarable

    // Remove specializations:
    bool Successful = VisitSpecializations(FTD);

    Successful &= VisitRedeclarableTemplateDecl(FTD);
    Successful &= VisitFunctionDecl(FTD->getTemplatedDecl());
    return Successful;
  }

  bool DeclUnloader::VisitClassTemplateDecl(ClassTemplateDecl* CTD) {
    // ClassTemplateDecl: TemplateDecl, Redeclarable

    if (CTD == m_Sema->StdInitializerList)
      m_Sema->StdInitializerList = m_Sema->StdInitializerList->getPreviousDecl();

    // Remove specializations:
    bool Successful = VisitSpecializations(CTD);

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

    const bool VisitingSpec = m_Flags & kVisitingSpecialization;
    const SourceLocation Loc = CTSD->getPointOfInstantiation();
    if (VisitingSpec && !Loc.isValid())
      return true;

    ClassTemplateSpecializationDecl* CanonCTSD =
      static_cast<ClassTemplateSpecializationDecl*>(CTSD->getCanonicalDecl());
    if (auto D = dyn_cast<ClassTemplatePartialSpecializationDecl>(CanonCTSD))
      ClassTemplateDeclExt::removePartialSpecialization(
                                                    D->getSpecializedTemplate(),
                                                    D);
    else
      ClassTemplateDeclExt::removeSpecialization(CTSD->getSpecializedTemplate(),
                                                 CanonCTSD);

    bool Success = true;
    if (VisitingSpec || !wasInstatiatedBefore(Loc)) {
      VisitorState VS(*this, kVisitingSpecialization);
      Success = VisitCXXRecordDecl(CTSD);
    }

    return Success;
  }
} // end namespace cling
