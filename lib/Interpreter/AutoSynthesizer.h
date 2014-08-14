//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_AUTO_SYNTHESIZER_H
#define CLING_AUTO_SYNTHESIZER_H

#include "TransactionTransformer.h"

#include <memory>

namespace clang {
  class Sema;
}

namespace llvm {
  class raw_ostream;
}

namespace cling {
  class AutoFixer;

  class AutoSynthesizer : public TransactionTransformer {
  private:
    std::unique_ptr<AutoFixer> m_AutoFixer;

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
