//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "DeclCollector.h"

#include "cling/Interpreter/Transaction.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"

using namespace clang;

namespace cling {

  // pin the vtable here.
  DeclCollector::~DeclCollector() {
  }

  bool DeclCollector::HandleTopLevelDecl(DeclGroupRef DGR) {
    m_CurTransaction->appendUnique(DGR);
    return true;
  }

  void DeclCollector::HandleInterestingDecl(DeclGroupRef DGR) {
    assert("Not implemented yet!");
  }

  void DeclCollector::HandleTagDeclDefinition(TagDecl* TD) {
    m_CurTransaction->appendUnique(DeclGroupRef(TD));
  }

  void DeclCollector::HandleVTable(CXXRecordDecl* RD,
                                     bool DefinitionRequired) {
    assert("Not implemented yet!");
  }

  void DeclCollector::CompleteTentativeDefinition(VarDecl* VD) {
    assert("Not implemented yet!");
  }

  void DeclCollector::HandleTranslationUnit(ASTContext& Ctx) {
    assert("Not implemented yet!");
  }
} // namespace cling
