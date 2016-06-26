//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "CheckEmptyTransactionTransformer.h"

#include "DeclUnloader.h"
#include "cling/Utils/AST.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"

using namespace clang;

namespace cling {
  ASTTransformer::Result CheckEmptyTransactionTransformer::Transform(Decl* D) {
    FunctionDecl* FD = cast<FunctionDecl>(D);
    assert(utils::Analyze::IsWrapper(FD) && "Expected wrapper");

    CompoundStmt* CS = cast<CompoundStmt>(FD->getBody());
    if (!CS->size() || (CS->size() == 1 && isa<NullStmt>(CS->body_back()))) {
      // This is an empty wrapper function. Get rid of it.
      // We know that it didn't end up in the EE by design.
      UnloadDecl(m_Sema, FD);

      return Result(0, true);
    }
    return Result(D, true);
  }
} // end namespace cling
