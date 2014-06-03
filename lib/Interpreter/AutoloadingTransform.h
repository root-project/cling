#ifndef CLING_AUTOLOADING_TRANSFORM_H
#define CLING_AUTOLOADING_TRANSFORM_H

#include "TransactionTransformer.h"

#include "llvm/ADT/OwningPtr.h"

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
    AutoloadingTransform(clang::Sema* S);

    virtual ~AutoloadingTransform();

    virtual void Transform();
  };

} // namespace cling

#endif //CLING_AUTOLOADING_TRANSFORM_H
