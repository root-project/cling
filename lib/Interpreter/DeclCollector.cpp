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
  static bool comesFromASTReader(DeclGroupRef DGR) {
    assert(!DGR.isNull() && "DeclGroupRef is Null!");
    // Take the first/only decl in the group.
    Decl* D = *DGR.begin();
    return D->isFromASTFile();
  }

  // pin the vtable here.
  DeclCollector::~DeclCollector() {
  }

  bool DeclCollector::HandleTopLevelDecl(DeclGroupRef DGR) {
    // if that decl comes from an AST File, i.e. PCH/PCM, no transaction needed
    // pipe it directly to codegen.
    if (comesFromASTReader(DGR) && m_CodeGen)
      return m_CodeGen->HandleTopLevelDecl(DGR);

    Transaction::DelayCallInfo DCI(DGR, Transaction::kCCIHandleTopLevelDecl);
    m_CurTransaction->append(DCI);
    return true;
  }

  void DeclCollector::HandleInterestingDecl(DeclGroupRef DGR) {
    // if that decl comes from an AST File, i.e. PCH/PCM, no transaction needed
    // pipe it directly to codegen.
    if (comesFromASTReader(DGR) && m_CodeGen)
      return (void)m_CodeGen->HandleTopLevelDecl(DGR);

    Transaction::DelayCallInfo DCI(DGR, Transaction::kCCIHandleInterestingDecl);
    m_CurTransaction->append(DCI);
  }

  void DeclCollector::HandleTagDeclDefinition(TagDecl* TD) {
    // if that decl comes from an AST File, i.e. PCH/PCM, no transaction needed
    // pipe it directly to codegen.
    if (comesFromASTReader(DeclGroupRef(TD)) && m_CodeGen)
      return m_CodeGen->HandleTagDeclDefinition(TD);

    Transaction::DelayCallInfo DCI(DeclGroupRef(TD), 
                                   Transaction::kCCIHandleTagDeclDefinition);
    m_CurTransaction->append(DCI);    
  }

  void DeclCollector::HandleVTable(CXXRecordDecl* RD, bool DefinitionRequired) {
    // if that decl comes from an AST File, i.e. PCH/PCM, no transaction needed
    // pipe it directly to codegen.
    if (comesFromASTReader(DeclGroupRef(RD)) && m_CodeGen)
      return m_CodeGen->HandleVTable(RD, DefinitionRequired);

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
    if (comesFromASTReader(DeclGroupRef(D)) && m_CodeGen)
      return m_CodeGen->HandleCXXImplicitFunctionInstantiation(D);

    Transaction::DelayCallInfo DCI(DeclGroupRef(D),
                                   Transaction::kCCIHandleCXXImplicitFunctionInstantiation);
    m_CurTransaction->append(DCI);
  }
  void DeclCollector::HandleCXXStaticMemberVarInstantiation(VarDecl *D) {
    // if that decl comes from an AST File, i.e. PCH/PCM, no transaction needed
    // pipe it directly to codegen.
    if (comesFromASTReader(DeclGroupRef(D)) && m_CodeGen)
      return m_CodeGen->HandleCXXStaticMemberVarInstantiation(D);

    Transaction::DelayCallInfo DCI(DeclGroupRef(D),
                                   Transaction::kCCIHandleCXXStaticMemberVarInstantiation);
    m_CurTransaction->append(DCI);
  }
  
} // namespace cling
