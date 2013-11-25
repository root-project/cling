//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: 437122046a46b828d7e11a9ab281b2f9c50f8aec $
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "CheckEmptyTransactionTransformer.h"

#include "cling/Interpreter/Transaction.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

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
        //FIXME: Replace with a invocation to the decl reverter.
        FD->getLexicalDeclContext()->removeDecl(FD);
      }
    }
  }
} // end namespace cling
