//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasielv@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/Transaction.h"

#include "cling/Utils/AST.h"

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
    // register the wrapper if any.
    if (DGR.isSingleDecl()) {
      if (FunctionDecl* FD = dyn_cast<FunctionDecl>(DGR.getSingleDecl()))
        if (utils::Analyze::IsWrapper(FD)) {
          assert(!m_WrapperFD && "Two wrappers in one transaction?");
          m_WrapperFD = FD;
        }
    }
    m_DeclQueue.push_back(DGR);
  }

  void Transaction::dump() const {
    if (!size())
      return;

    ASTContext& C = getFirstDecl().getSingleDecl()->getASTContext();
    PrintingPolicy Policy = C.getPrintingPolicy();
    Policy.DumpSourceManager = &C.getSourceManager();
    print(llvm::outs(), Policy, /*Indent*/0, /*PrintInstantiation*/true);
  }

  void Transaction::dumpPretty() const {
    if (!size())
      return;
    ASTContext* C = 0;
    if (m_WrapperFD)
      C = &(m_WrapperFD->getASTContext());
    if (!getFirstDecl().isNull())
      C = &(getFirstDecl().getSingleDecl()->getASTContext());
      
    PrintingPolicy Policy(C->getLangOpts());
    print(llvm::outs(), Policy, /*Indent*/0, /*PrintInstantiation*/true);
  }

  void Transaction::print(llvm::raw_ostream& Out, const PrintingPolicy& Policy,
                          unsigned Indent, bool PrintInstantiation) const {
    int nestedT = 0;
    for (const_iterator I = decls_begin(), E = decls_end(); I != E; ++I) {
      if (I->isNull()) {
        assert(hasNestedTransactions() && "DGR is null even if no nesting?");
        // print the nested decl
        Out<< "\n";
        Out<<"+====================================================+\n";
        Out<<"        Nested Transaction" << nestedT << "           \n";
        Out<<"+====================================================+\n";
        m_NestedTransactions[nestedT++]->print(Out, Policy, Indent, 
                                               PrintInstantiation);
        Out<< "\n";
        Out<<"+====================================================+\n";
        Out<<"          End Transaction" << nestedT << "            \n";
        Out<<"+====================================================+\n";
      }
      for (DeclGroupRef::const_iterator J = I->begin(), L = I->end();J != L;++J)
        if (*J)
          (*J)->print(Out, Policy, Indent, PrintInstantiation);
        else
          Out << "<<NULL DECL>>";
    }
  }

} // end namespace cling
