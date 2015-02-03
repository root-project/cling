//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "CheckEmptyTransactionTransformer.h"

#include "TransactionUnloader.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Sema/Sema.h"

#include <algorithm>

using namespace clang;

namespace cling {
  void CheckEmptyTransactionTransformer::Transform() {
    Transaction* T = getTransaction();
    if (FunctionDecl* FD = T->getWrapperFD()) {
      CompoundStmt* CS = cast<CompoundStmt>(FD->getBody());
      if (!CS->size() || (CS->size() == 1 && isa<NullStmt>(CS->body_back()))) {
        // This is an empty wrapper function. Get rid of it.
        DeclGroupRef DGR(FD);
        Transaction::DelayCallInfo DCI (DGR,
                                        Transaction::kCCIHandleTopLevelDecl);
        Transaction::iterator found
          = std::find(T->decls_begin(), T->decls_end(), DCI);
        if (found != T->decls_end()) {
          T->erase(found);
        }
        // We know that it didn't end up in the EE by design.
        TransactionUnloader U(m_Sema, /*CodeGenerator*/0);
        U.UnloadDecl(FD);
      }
    }
  }
} // end namespace cling
