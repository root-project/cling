//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasielv@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/Transaction.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/PrettyPrinter.h"

using namespace clang;

namespace cling {

  Transaction::~Transaction() {
    for (size_t i = 0; i < m_NestedTransactions.size(); ++i)
      delete m_NestedTransactions[i];
  }

  void Transaction::appendUnique(DeclGroupRef DGR) {
    for (const_iterator I = decls_begin(), E = decls_end(); I != E; ++I) {
      if (DGR.isNull() || (*I).getAsOpaquePtr() == DGR.getAsOpaquePtr())
        return;
    }
    m_DeclQueue.push_back(DGR);
  }

  void Transaction::dump() const {
    ASTContext& C = getFirstDecl().getSingleDecl()->getASTContext();
    PrintingPolicy Policy = C.getPrintingPolicy();
    Policy.DumpSourceManager = &C.getSourceManager();
    print(llvm::outs(), Policy, /*Indent*/0, /*PrintInstantiation*/true);
  }

  void Transaction::dumpPretty() const {
    ASTContext& C = getFirstDecl().getSingleDecl()->getASTContext();
    PrintingPolicy Policy(C.getLangOpts());
    print(llvm::outs(), Policy, /*Indent*/0, /*PrintInstantiation*/true);
  }

  void Transaction::print(llvm::raw_ostream& Out, const PrintingPolicy& Policy,
                          unsigned Indent, bool PrintInstantiation) const {
    int nestedT = 0;
    for (const_iterator I = decls_begin(), E = decls_end(); I != E; ++I) {
      if (I->isNull()) {
        assert(hasNestedTransactions() && "DGR is null even if no nesting?");
        // print the nested decl
        Out<<"+====================================================+\n";
        Out<<"|       Nested Transaction" << nestedT << "          |\n";
        Out<<"+====================================================+\n";
        m_NestedTransactions[nestedT++]->print(Out, Policy, Indent, 
                                               PrintInstantiation);
        Out<<"+====================================================+\n";
        Out<<"|         End Transaction" << nestedT << "           |\n";
        Out<<"+====================================================+\n";
      }
      for (DeclGroupRef::const_iterator J = I->begin(), L = I->end(); J != L;++J)
        (*J)->print(Out, Policy, Indent, PrintInstantiation);
    }
  }

} // end namespace cling
