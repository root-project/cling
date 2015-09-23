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
#include "clang/Lex/Token.h"

using namespace clang;

namespace {
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
}

namespace cling {
  bool DeclCollector::comesFromASTReader(DeclGroupRef DGR) const {
    assert(!DGR.isNull() && "DeclGroupRef is Null!");
    assert(m_CurTransaction && "No current transaction when deserializing");
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

  void DeclCollector::AddedCXXImplicitMember(const CXXRecordDecl *RD,
                                             const Decl *D) {
    assert(D->isImplicit());
    // We need to mark the decls coming from the modules
    if (comesFromASTReader(RD) || comesFromASTReader(D)) {
      Decl* implicitD = const_cast<Decl*>(D);
      implicitD->addAttr(UsedAttr::CreateImplicit(implicitD->getASTContext()));
    }
  }

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

  bool DeclCollector::Transform(DeclGroupRef& DGR) const {
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

    Transaction::DelayCallInfo DCI(DGR, Transaction::kCCIHandleTopLevelDecl);
    m_CurTransaction->append(DCI);
    if (!m_Consumer
        || getTransaction()->getIssuedDiags() == Transaction::kErrors)
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
    Transaction::DelayCallInfo DCI(DGR, Transaction::kCCIHandleInterestingDecl);
    m_CurTransaction->append(DCI);
    if (m_Consumer
        && (!comesFromASTReader(DGR) || !shouldIgnore(*DGR.begin())))
      m_Consumer->HandleTopLevelDecl(DGR);
  }

  void DeclCollector::HandleTagDeclDefinition(TagDecl* TD) {
    Transaction::DelayCallInfo DCI(DeclGroupRef(TD),
                                   Transaction::kCCIHandleTagDeclDefinition);
    m_CurTransaction->append(DCI);
    if (m_Consumer
        && (!comesFromASTReader(DeclGroupRef(TD))
            || !shouldIgnore(TD)))
      m_Consumer->HandleTagDeclDefinition(TD);
  }

  void DeclCollector::HandleInvalidTagDeclDefinition(clang::TagDecl *TD){
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
    Transaction::DelayCallInfo DCI(DeclGroupRef(D),
                                   Transaction::kCCIHandleCXXImplicitFunctionInstantiation);
    m_CurTransaction->append(DCI);
    if (m_Consumer
        && (!comesFromASTReader(DeclGroupRef(D))
            || !shouldIgnore(D)))
    m_Consumer->HandleCXXImplicitFunctionInstantiation(D);
  }

  void DeclCollector::HandleCXXStaticMemberVarInstantiation(VarDecl *D) {
    Transaction::DelayCallInfo DCI(DeclGroupRef(D),
                                   Transaction::kCCIHandleCXXStaticMemberVarInstantiation);
    m_CurTransaction->append(DCI);
    if (m_Consumer
        && (!comesFromASTReader(DeclGroupRef(D))
            || !shouldIgnore(D)))
    m_Consumer->HandleCXXStaticMemberVarInstantiation(D);
  }

  void DeclCollector::MacroDefined(const clang::Token &MacroNameTok,
                                   const clang::MacroDirective *MD) {
    Transaction::MacroDirectiveInfo MDE(MacroNameTok.getIdentifierInfo(), MD);
    m_CurTransaction->append(MDE);
  }

} // namespace cling
