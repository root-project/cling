//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_RETURN_SYNTHESIZER_H
#define CLING_RETURN_SYNTHESIZER_H

#include "TransactionTransformer.h"

namespace clang {
  class ASTContext;
  class Sema;
}

namespace llvm {
  class raw_ostream;
}

namespace cling {

  class ReturnSynthesizer : public TransactionTransformer {

  private:
    ///\brief Needed for the AST transformations, owned by Sema.
    ///
    clang::ASTContext* m_Context;

public:
    ///\ brief Constructs the return synthesizer.
    ///
    ///\param[in] S - The semantic analysis object.
    ///
    ReturnSynthesizer(clang::Sema* S);
    
    virtual ~ReturnSynthesizer();

    virtual void Transform();
  };

} // namespace cling

#endif // CLING_RETURN_SYNTHESIZER_H
