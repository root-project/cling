//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: 27217adceba602502bdfdd8a756fbbc017f275fa $
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
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
    ASTDumper() : TransactionTransformer(0) { }
    virtual ~ASTDumper();

    virtual void Transform();

  private:
    void printDecl(clang::Decl* D);
  };

} // namespace cling

#endif // CLING_AST_DUMPER_H
