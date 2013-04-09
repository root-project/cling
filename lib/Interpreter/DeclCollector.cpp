//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "DeclCollector.h"

#include "cling/Interpreter/Transaction.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"

#include "clang/CodeGen/ModuleBuilder.h"

using namespace clang;

namespace cling {
  bool DeclCollector::comesFromASTReader(DeclGroupRef DGR) const {
    assert(!DGR.isNull() && "DeclGroupRef is Null!");
    if (m_CurTransaction->getCompilationOpts().CodeGenerationForModule)
      return true;

    // Take the first/only decl in the group.
    Decl* D = *DGR.begin();
    return D->isFromASTFile();
  }

  bool DeclCollector::shouldIgnoreDeclFromASTReader(const Decl* D) const {
    // Functions that are inlined must be sent to CodeGen - they will not have a
    // symbol in the library.
    if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(D))
      return !FD->isInlined();

    // Don't codegen statics coming in from a module; they are already part of
    // the library.
    if (const VarDecl* VD = dyn_cast<VarDecl>(D))
      if (VD->hasGlobalStorage())
        return true;
    return false;
  }

  // pin the vtable here.
  DeclCollector::~DeclCollector() {
  }

  bool DeclCollector::HandleTopLevelDecl(DeclGroupRef DGR) {
    // if that decl comes from an AST File, i.e. PCH/PCM, no transaction needed
    // pipe it directly to codegen.
    if (comesFromASTReader(DGR)) {
      if (m_CodeGen) {
        for (DeclGroupRef::iterator I = DGR.begin(), E = DGR.end();
             I != E; ++I)
          if (NamespaceDecl* ND = dyn_cast<NamespaceDecl>(*I)) {
            for (NamespaceDecl::decl_iterator IN = ND->decls_begin(),
                   EN = ND->decls_end(); IN != EN; ++IN)
              if (!shouldIgnoreDeclFromASTReader(*IN))
                m_CodeGen->HandleTopLevelDecl(DeclGroupRef(*IN));
          } else if (!shouldIgnoreDeclFromASTReader(*I))
            m_CodeGen->HandleTopLevelDecl(DeclGroupRef(*I));
      }
      return true;
    }

    Transaction::DelayCallInfo DCI(DGR, Transaction::kCCIHandleTopLevelDecl);
    m_CurTransaction->append(DCI);
    return true;
  }

  void DeclCollector::HandleInterestingDecl(DeclGroupRef DGR) {
    // if that decl comes from an AST File, i.e. PCH/PCM, no transaction needed
    // pipe it directly to codegen.
    if (comesFromASTReader(DGR)) {
      if (m_CodeGen) {
        for (DeclGroupRef::iterator I = DGR.begin(), E = DGR.end();
             I != E; ++I)
          if (!shouldIgnoreDeclFromASTReader(*I))
            m_CodeGen->HandleTopLevelDecl(DeclGroupRef(*I));
      }
      return;
    }

    Transaction::DelayCallInfo DCI(DGR, Transaction::kCCIHandleInterestingDecl);
    m_CurTransaction->append(DCI);
  }

  void DeclCollector::HandleTagDeclDefinition(TagDecl* TD) {
    // if that decl comes from an AST File, i.e. PCH/PCM, no transaction needed
    // pipe it directly to codegen.
    if (comesFromASTReader(DeclGroupRef(TD))) {
      if (m_CodeGen)
        m_CodeGen->HandleTagDeclDefinition(TD);
      return;
    }

    Transaction::DelayCallInfo DCI(DeclGroupRef(TD), 
                                   Transaction::kCCIHandleTagDeclDefinition);
    m_CurTransaction->append(DCI);    
  }

  void DeclCollector::HandleVTable(CXXRecordDecl* RD, bool DefinitionRequired) {
    // if that decl comes from an AST File, i.e. PCH/PCM, no transaction needed
    // pipe it directly to codegen.
    if (comesFromASTReader(DeclGroupRef(RD))) {
      // FIXME: when is the vtable part of the library?
      if (m_CodeGen)
        m_CodeGen->HandleVTable(RD, DefinitionRequired);
      return;
    }

    Transaction::DelayCallInfo DCI(DeclGroupRef(RD),
                                   Transaction::kCCIHandleVTable);
    m_CurTransaction->append(DCI);    

    // Intentional no-op. It comes through Sema::DefineUsedVTables, which
    // comes either Sema::ActOnEndOfTranslationUnit or while instantiating a
    // template. In our case we will do it on transaction commit, without 
    // keeping track of used vtables, because we have cases where we bypass the
    // clang/AST and directly ask the module so that we have to generate 
    // everything without extra smartness.
  }

  void DeclCollector::CompleteTentativeDefinition(VarDecl* VD) {
    assert(0 && "Not implemented yet!");
  }

  void DeclCollector::HandleTranslationUnit(ASTContext& Ctx) {
  }

  void DeclCollector::HandleCXXImplicitFunctionInstantiation(FunctionDecl *D) {
    // if that decl comes from an AST File, i.e. PCH/PCM, no transaction needed
    // pipe it directly to codegen.
    if (comesFromASTReader(DeclGroupRef(D))) {
      if (m_CodeGen)
        m_CodeGen->HandleCXXImplicitFunctionInstantiation(D);
      return;
    }

    Transaction::DelayCallInfo DCI(DeclGroupRef(D),
                                   Transaction::kCCIHandleCXXImplicitFunctionInstantiation);
    m_CurTransaction->append(DCI);
  }
  void DeclCollector::HandleCXXStaticMemberVarInstantiation(VarDecl *D) {
    // if that decl comes from an AST File, i.e. PCH/PCM, no transaction needed
    // pipe it directly to codegen.
    if (comesFromASTReader(DeclGroupRef(D))) {
      if (m_CodeGen && !shouldIgnoreDeclFromASTReader(D))
          m_CodeGen->HandleCXXStaticMemberVarInstantiation(D);
      return;
    }

    Transaction::DelayCallInfo DCI(DeclGroupRef(D),
                                   Transaction::kCCIHandleCXXStaticMemberVarInstantiation);
    m_CurTransaction->append(DCI);
  }
  
} // namespace cling
