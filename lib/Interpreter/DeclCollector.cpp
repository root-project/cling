//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "DeclCollector.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Lex/Token.h"

using namespace clang;

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

  bool DeclCollector::HandleTopLevelDecl(DeclGroupRef DGR) {
    Transaction::DelayCallInfo DCI(DGR, Transaction::kCCIHandleTopLevelDecl);
    m_CurTransaction->append(DCI);
    return true;
  }

  void DeclCollector::HandleInterestingDecl(DeclGroupRef DGR) {
    Transaction::DelayCallInfo DCI(DGR, Transaction::kCCIHandleInterestingDecl);
    m_CurTransaction->append(DCI);
  }

  void DeclCollector::HandleTagDeclDefinition(TagDecl* TD) {
    Transaction::DelayCallInfo DCI(DeclGroupRef(TD),
                                   Transaction::kCCIHandleTagDeclDefinition);
    m_CurTransaction->append(DCI);
  }

  void DeclCollector::HandleVTable(CXXRecordDecl* RD) {
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
    // C has tentative definitions which we might need to deal with when running
    // in C mode.
    Transaction::DelayCallInfo DCI(DeclGroupRef(VD),
                                   Transaction::kCCICompleteTentativeDefinition);
    m_CurTransaction->append(DCI);
  }

  void DeclCollector::HandleTranslationUnit(ASTContext& Ctx) {
  }

  void DeclCollector::HandleCXXImplicitFunctionInstantiation(FunctionDecl *D) {
    Transaction::DelayCallInfo DCI(DeclGroupRef(D),
                                   Transaction::kCCIHandleCXXImplicitFunctionInstantiation);
    m_CurTransaction->append(DCI);
  }
  void DeclCollector::HandleCXXStaticMemberVarInstantiation(VarDecl *D) {
    Transaction::DelayCallInfo DCI(DeclGroupRef(D),
                                   Transaction::kCCIHandleCXXStaticMemberVarInstantiation);
    m_CurTransaction->append(DCI);
  }

  void DeclCollector::MacroDefined(const clang::Token &MacroNameTok,
                                   const clang::MacroDirective *MD) {
    Transaction::MacroDirectiveInfo MDE(MacroNameTok.getIdentifierInfo(), MD);
    m_CurTransaction->append(MDE);
  }

} // namespace cling
