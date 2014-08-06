#ifndef CLING_AUTOLOADING_TRANSFORM_H
#define CLING_AUTOLOADING_TRANSFORM_H

#include "TransactionTransformer.h"

namespace clang {
  class Sema;
}

namespace cling {

  class AutoloadingTransform : public TransactionTransformer {
  public:
    ///\ brief Constructs the auto synthesizer.
    ///
    ///\param[in] S - The semantic analysis object.
    ///
    AutoloadingTransform(clang::Sema* S) : TransactionTransformer(S) {}

    virtual void Transform();
  };

} // namespace cling

#endif //CLING_AUTOLOADING_TRANSFORM_H
