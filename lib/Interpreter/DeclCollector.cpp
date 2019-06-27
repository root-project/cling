//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "DeclCollector.h"

#include "IncrementalParser.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "llvm/Support/Signals.h"

using namespace clang;

namespace {
  ///\brief Return true if this decl (which comes from an AST file) should
  /// not be sent to CodeGen. The module is assumed to describe the contents
  /// of a library; symbols inside the library must thus not be reemitted /
  /// duplicated by CodeGen.
  ///
  static bool shouldIgnore(const Decl* D) {
    // This function is called for all "deserialized" decls, where the
    // "deserialized" decl either really comes from an AST file or from
    // a header that's loaded to import the AST for a library with a dictionary
    // (the non-PCM case).
    //
    // Functions that are inlined must be sent to CodeGen - they will not have a
    // symbol in the library.
    if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(D)) {
      if (D->isFromASTFile()) {
        return !FD->hasBody();
      } else {
        // If the decl must be emitted then it will be in the library.
        // If not, we must expose it to CodeGen now because it might
        // not be in the library. Does this correspond to a weak symbol
        // by definition?
        return !(FD->isInlined() || FD->isTemplateInstantiation());
      }
    }
    // Don't codegen statics coming in from a module; they are already part of
    // the library.
    // We do need to expose static variables from template instantiations.
    if (const VarDecl* VD = dyn_cast<VarDecl>(D))
      if (VD->hasGlobalStorage() && !VD->getType().isConstQualified()
          && VD->getTemplateSpecializationKind() == TSK_Undeclared)
        return true;
    return false;
  }

  /// \brief Asserts that the given transaction is not null, otherwise prints a
  /// stack trace to stderr and aborts execution.
  static void assertHasTransaction(const cling::Transaction* T) {
    if (!T) {
      llvm::sys::PrintStackTrace(llvm::errs());
      llvm_unreachable("Missing transaction during deserialization!");
    }
  }
}

namespace cling {
  ///\brief Serves as DeclCollector's connector to the PPCallbacks interface.
  ///
  class DeclCollector::PPAdapter : public clang::PPCallbacks {
    cling::DeclCollector* m_Parent;

    void MacroDirective(const clang::Token& MacroNameTok,
                        const clang::MacroDirective* MD) {
      assertHasTransaction(m_Parent->m_CurTransaction);
      Transaction::MacroDirectiveInfo MDE(MacroNameTok.getIdentifierInfo(), MD);
      m_Parent->m_CurTransaction->append(MDE);
    }

  public:
    PPAdapter(cling::DeclCollector* P) : m_Parent(P) {}

    /// \name PPCallbacks overrides
    /// Macro support
    void MacroDefined(const clang::Token& MacroNameTok,
                      const clang::MacroDirective* MD) final {
      MacroDirective(MacroNameTok, MD);
    }

    /// \name PPCallbacks overrides
    /// Macro support
    void MacroUndefined(const clang::Token& MacroNameTok,
                        const clang::MacroDefinition& MD,
                        const clang::MacroDirective* Undef) final {
      if (Undef)
        MacroDirective(MacroNameTok, Undef);
    }
  };

  void DeclCollector::Setup(IncrementalParser* IncrParser,
                            std::unique_ptr<ASTConsumer> Consumer,
                            clang::Preprocessor& PP) {
    m_IncrParser = IncrParser;
    m_Consumer = std::move(Consumer);
    PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(new PPAdapter(this)));
  }

  bool DeclCollector::comesFromASTReader(DeclGroupRef DGR) const {
    assert(!DGR.isNull() && "DeclGroupRef is Null!");
    assertHasTransaction(m_CurTransaction);
    if (m_CurTransaction->getCompilationOpts().CodeGenerationForModule)
      return true;

    // Take the first/only decl in the group.
    Decl* D = *DGR.begin();
    return D->isFromASTFile();
  }

  bool DeclCollector::comesFromASTReader(const Decl* D) const {
    // The operation is const but clang::DeclGroupRef doesn't allow us to
    // express it.
    return comesFromASTReader(DeclGroupRef(const_cast<Decl*>(D)));
  }

  // pin the vtable here.
  DeclCollector::~DeclCollector() { }

 ASTTransformer::Result DeclCollector::TransformDecl(Decl* D) const {
    // We are sure it's safe to pipe it through the transformers
    // Consume late transformers init
    for (size_t i = 0; D && i < m_TransactionTransformers.size(); ++i) {
      ASTTransformer::Result NewDecl
        = m_TransactionTransformers[i]->Transform(D, m_CurTransaction);
      if (!NewDecl.getInt()) {
        m_CurTransaction->setIssuedDiags(Transaction::kErrors);
        return NewDecl;
      }
      D = NewDecl.getPointer();
    }
    if (FunctionDecl* FD = dyn_cast_or_null<FunctionDecl>(D)) {
      if (utils::Analyze::IsWrapper(FD)) {
        for (size_t i = 0; D && i < m_WrapperTransformers.size(); ++i) {
          ASTTransformer::Result NewDecl
           = m_WrapperTransformers[i]->Transform(D, m_CurTransaction);
          if (!NewDecl.getInt()) {
            m_CurTransaction->setIssuedDiags(Transaction::kErrors);
            return NewDecl;
          }
          D = NewDecl.getPointer();
        }
      }
    }
    return ASTTransformer::Result(D, true);
  }

  bool DeclCollector::Transform(DeclGroupRef& DGR) {
    // Do not tranform recursively, e.g. when emitting a DeclExtracted decl.
    if (m_Transforming)
      return true;

    struct TransformingRAII {
      bool& m_Transforming;
      TransformingRAII(bool& Transforming): m_Transforming(Transforming) {
        m_Transforming = true;
      }
      ~TransformingRAII() { m_Transforming = false; }
    } transformingUpdater(m_Transforming);

    llvm::SmallVector<Decl*, 4> ReplacedDecls;
    bool HaveReplacement = false;
    for (Decl* D: DGR) {
      ASTTransformer::Result NewDecl = TransformDecl(D);
      if (!NewDecl.getInt())
        return false;
      HaveReplacement |= (NewDecl.getPointer() != D);
      if (NewDecl.getPointer())
        ReplacedDecls.push_back(NewDecl.getPointer());
    }
    if (HaveReplacement)
      DGR = DeclGroupRef::Create((*DGR.begin())->getASTContext(),
                                 ReplacedDecls.data(), ReplacedDecls.size());
    return true;
  }

  bool DeclCollector::HandleTopLevelDecl(DeclGroupRef DGR) {
    if (!Transform(DGR))
      return false;

    if (DGR.isNull())
      return true;

    if (!m_Consumer)
      return true;

    assertHasTransaction(m_CurTransaction);

    Transaction::DelayCallInfo DCI(DGR, Transaction::kCCIHandleTopLevelDecl);
    m_CurTransaction->append(DCI);

    if (getTransaction()->getIssuedDiags() == Transaction::kErrors)
      return true;

    if (comesFromASTReader(DGR)) {
      for (DeclGroupRef::iterator DI = DGR.begin(), DE = DGR.end();
           DI != DE; ++DI) {
        DeclGroupRef SplitDGR(*DI);
        // FIXME: The special namespace treatment (not sending itself to
        // CodeGen, but only its content - if the contained decl should be
        // emitted) works around issue with the static initialization when
        // having a PCH and loading a library. We don't want to generate
        // code for the static that will come through the library.
        //
        // This will be fixed with the clang::Modules. Make sure we remember.
        // assert(!getCI()->getLangOpts().Modules && "Please revisit!");
        if (NamespaceDecl* ND = dyn_cast<NamespaceDecl>(*DI)) {
          for (NamespaceDecl::decl_iterator NDI = ND->decls_begin(),
               EN = ND->decls_end(); NDI != EN; ++NDI) {
            // Recurse over decls inside the namespace, like
            // CodeGenModule::EmitNamespace() does.
            if (!shouldIgnore(*NDI))
              m_Consumer->HandleTopLevelDecl(DeclGroupRef(*NDI));
          }
        } else if (!shouldIgnore(*DI)) {
          m_Consumer->HandleTopLevelDecl(DeclGroupRef(*DI));
        }
        continue;
      }
    } else {
      m_Consumer->HandleTopLevelDecl(DGR);
    }
    return true;
  }

  void DeclCollector::HandleInterestingDecl(DeclGroupRef DGR) {
    assertHasTransaction(m_CurTransaction);
    Transaction::DelayCallInfo DCI(DGR, Transaction::kCCIHandleInterestingDecl);
    m_CurTransaction->append(DCI);
    if (m_Consumer
        && (!comesFromASTReader(DGR) || !shouldIgnore(*DGR.begin())))
      m_Consumer->HandleTopLevelDecl(DGR);
  }

  void DeclCollector::HandleTagDeclDefinition(TagDecl* TD) {
    assertHasTransaction(m_CurTransaction);
    Transaction::DelayCallInfo DCI(DeclGroupRef(TD),
                                   Transaction::kCCIHandleTagDeclDefinition);
    m_CurTransaction->append(DCI);
    if (m_Consumer
        && (!comesFromASTReader(DeclGroupRef(TD))
            || !shouldIgnore(TD)))
      m_Consumer->HandleTagDeclDefinition(TD);
  }

  void DeclCollector::HandleInvalidTagDeclDefinition(clang::TagDecl *TD){
    assertHasTransaction(m_CurTransaction);
    Transaction::DelayCallInfo DCI(DeclGroupRef(TD),
                                   Transaction::kCCIHandleTagDeclDefinition);
    m_CurTransaction->append(DCI);
    m_CurTransaction->setIssuedDiags(Transaction::kErrors);
    if (m_Consumer
        && (!comesFromASTReader(DeclGroupRef(TD))
            || !shouldIgnore(TD)))
      m_Consumer->HandleInvalidTagDeclDefinition(TD);
  }

  void DeclCollector::HandleVTable(CXXRecordDecl* RD) {
    assertHasTransaction(m_CurTransaction);
    Transaction::DelayCallInfo DCI(DeclGroupRef(RD),
                                   Transaction::kCCIHandleVTable);
    m_CurTransaction->append(DCI);

    if (m_Consumer
        && (!comesFromASTReader(DeclGroupRef(RD))
            || !shouldIgnore(RD)))
      m_Consumer->HandleVTable(RD);
    // Intentional no-op. It comes through Sema::DefineUsedVTables, which
    // comes either Sema::ActOnEndOfTranslationUnit or while instantiating a
    // template. In our case we will do it on transaction commit, without
    // keeping track of used vtables, because we have cases where we bypass the
    // clang/AST and directly ask the module so that we have to generate
    // everything without extra smartness.
  }

  void DeclCollector::CompleteTentativeDefinition(VarDecl* VD) {
    assertHasTransaction(m_CurTransaction);
    // C has tentative definitions which we might need to deal with when running
    // in C mode.
    Transaction::DelayCallInfo DCI(DeclGroupRef(VD),
                                   Transaction::kCCICompleteTentativeDefinition);
    m_CurTransaction->append(DCI);
    if (m_Consumer
        && (!comesFromASTReader(DeclGroupRef(VD))
            || !shouldIgnore(VD)))
    m_Consumer->CompleteTentativeDefinition(VD);
  }

  void DeclCollector::HandleTranslationUnit(ASTContext& Ctx) {
    //if (m_Consumer)
    //  m_Consumer->HandleTranslationUnit(Ctx);
  }

  void DeclCollector::HandleCXXImplicitFunctionInstantiation(FunctionDecl *D) {
    assertHasTransaction(m_CurTransaction);
    Transaction::DelayCallInfo DCI(DeclGroupRef(D),
                                   Transaction::kCCIHandleCXXImplicitFunctionInstantiation);
    m_CurTransaction->append(DCI);
    if (m_Consumer
        && (!comesFromASTReader(DeclGroupRef(D))
            || !shouldIgnore(D)))
    m_Consumer->HandleCXXImplicitFunctionInstantiation(D);
  }

  void DeclCollector::HandleCXXStaticMemberVarInstantiation(VarDecl *D) {
    assertHasTransaction(m_CurTransaction);
    Transaction::DelayCallInfo DCI(DeclGroupRef(D),
                                   Transaction::kCCIHandleCXXStaticMemberVarInstantiation);
    m_CurTransaction->append(DCI);
    if (m_Consumer
        && (!comesFromASTReader(DeclGroupRef(D))
            || !shouldIgnore(D)))
    m_Consumer->HandleCXXStaticMemberVarInstantiation(D);
  }

} // namespace cling
