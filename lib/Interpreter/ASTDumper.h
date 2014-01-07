//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_AST_DUMPER_H
#define CLING_AST_DUMPER_H

#include "TransactionTransformer.h"

namespace clang {
  class Decl;
}

namespace cling {

  class Transaction;

  // TODO : This is not really a transformer. Factor out.
  class ASTDumper : public TransactionTransformer {
  public:
    ASTDumper() : TransactionTransformer(/*Sema=*/0) { }
    virtual ~ASTDumper();

    virtual void Transform();

  private:
    void printDecl(clang::Decl* D);
  };

} // namespace cling

#endif // CLING_AST_DUMPER_H
