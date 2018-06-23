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

#include "llvm/ADT/DenseMap.h"

namespace clang {
  class Decl;
  class DirectoryEntry;
}
namespace cling {
  class Interpreter;
}

namespace cling {

  class NullDerefProtectionTransformer : public ASTTransformer {
    cling::Interpreter* m_Interp;

    /// Whether to visit a Decl coming from a file in a given directory.
    llvm::DenseMap<const clang::DirectoryEntry*, bool> m_ShouldVisitDir;

    /// Whether the declaration should be visited and possibly transformed.
    bool shouldTransform(const clang::Decl* D);

  public:
    ///\ brief Constructs the NullDeref AST Transformer.
    ///
    ///\param[in] I - The interpreter.
    ///
    NullDerefProtectionTransformer(cling::Interpreter* I);

    virtual ~NullDerefProtectionTransformer();
    Result Transform(clang::Decl* D) override;
  };

} // namespace cling

#endif // CLING_AST_NULL_DEREF_PROTECTION_H
