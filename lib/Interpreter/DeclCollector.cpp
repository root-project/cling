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

  // Does more than we want:
  // if there is class A {enum E {kEnum = 1};};
  // we get two different tag decls one for A and one for E. This is not that 
  // bad because esentially it has no effect on codegen but it differs from what
  // one'd expect. For now rely on the HandleTopLevelDecl to provide all the 
  // declarations in the transaction.
  void DeclCollector::HandleTagDeclDefinition(TagDecl* TD) {
    // Intentional no-op.
  }

  void DeclCollector::HandleVTable(CXXRecordDecl* RD, bool DefinitionRequired) {
    assert(0 && "Not implemented yet!");
  }

  void DeclCollector::CompleteTentativeDefinition(VarDecl* VD) {
    assert(0 && "Not implemented yet!");
  }

  void DeclCollector::HandleTranslationUnit(ASTContext& Ctx) {
  }
} // namespace cling
