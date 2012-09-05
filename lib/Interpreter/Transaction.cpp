//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasielv@cern.ch>
//------------------------------------------------------------------------------

#include "Transaction.h"

#include "clang/AST/DeclBase.h"

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
    int nestedT = 0;
    for (const_iterator I = decls_begin(), E = decls_end(); I != E; ++I) {
      if (I->isNull()) {
        assert(hasNestedTransactions() && "DGR is null even if no nesting?");
        // print the nested decl
        llvm::outs()<<"+====================================================+\n";
        llvm::outs()<<"|       Nested Transaction" << nestedT << "          |\n";
        llvm::outs()<<"+====================================================+\n";
        m_NestedTransactions[nestedT++]->dump();        
        llvm::outs()<<"+====================================================+\n";
        llvm::outs()<<"|         End Transaction" << nestedT << "           |\n";
        llvm::outs()<<"+====================================================+\n";
      }
      for (DeclGroupRef::const_iterator J = I->begin(), L = I->end(); J != L;++J)
        (*J)->dump();
    }
  }

} // end namespace cling
