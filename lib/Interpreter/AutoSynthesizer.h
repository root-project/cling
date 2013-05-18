//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id: cb7241880ebcbba87b2ae16476c2812afd7ff571 $
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_AUTO_SYNTHESIZER_H
#define CLING_AUTO_SYNTHESIZER_H

#include "TransactionTransformer.h"

namespace clang {
  class Sema;
}

namespace llvm {
  class raw_ostream;
}

namespace cling {

  class AutoSynthesizer : public TransactionTransformer {

public:
    ///\ brief Constructs the auto synthesizer.
    ///
    ///\param[in] S - The semantic analysis object.
    ///
    AutoSynthesizer(clang::Sema* S);
    
    virtual ~AutoSynthesizer();

    virtual void Transform();
  };

} // namespace cling

#endif // CLING_AUTO_SYNTHESIZER_H
