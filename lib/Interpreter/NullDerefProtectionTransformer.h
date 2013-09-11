//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: cb7241880ebcbba87b2ae16476c2812afd7ff571 $
// author:  Baozeng Ding <sploving1@gmail.com>
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_AST_NULL_DEREF_PROTECTION_H
#define CLING_AST_NULL_DEREF_PROTECTION_H

#include "TransactionTransformer.h"

namespace clang {
  class Sema;
}

namespace cling {

  class NullDerefProtectionTransformer : public TransactionTransformer {

   
    public:
      ///\ brief Constructs the NullDeref AST Transformer.
      ///
      ///\param[in] S - The semantic analysis object.
      ///
      NullDerefProtectionTransformer(clang::Sema* S);

      virtual ~NullDerefProtectionTransformer();
      virtual void Transform();
  };

} // namespace cling

#endif // CLING_AST_NULL_DEREF_PROTECTION_H
