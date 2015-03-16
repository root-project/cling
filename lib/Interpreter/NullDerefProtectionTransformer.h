//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Baozeng Ding <sploving1@gmail.com>
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_AST_NULL_DEREF_PROTECTION_H
#define CLING_AST_NULL_DEREF_PROTECTION_H

#include "ASTTransformer.h"

namespace clang {
  class Decl;
  class Sema;
}

namespace cling {

  class NullDerefProtectionTransformer : public WrapperTransformer {
  public:
    ///\ brief Constructs the NullDeref AST Transformer.
    ///
    ///\param[in] S - The semantic analysis object.
    ///
    NullDerefProtectionTransformer(clang::Sema* S);

    virtual ~NullDerefProtectionTransformer();
    Result Transform(clang::Decl* D) override;
  };

} // namespace cling

#endif // CLING_AST_NULL_DEREF_PROTECTION_H
