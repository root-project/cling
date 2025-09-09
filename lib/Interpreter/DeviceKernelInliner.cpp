//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "DeviceKernelInliner.h"

#ifdef _MSC_VER
// FIXME: Needed on Windows after LLVM 20 update to build
#include <clang/AST/ASTContext.h>
#endif
#include <clang/AST/Attr.h>

#include <llvm/Support/Casting.h>

namespace cling {

DeviceKernelInliner::DeviceKernelInliner(clang::Sema *S) : ASTTransformer(S) {}

ASTTransformer::Result DeviceKernelInliner::Transform(clang::Decl *D) {
  if (clang::FunctionDecl* F = llvm::dyn_cast<clang::FunctionDecl>(D)) {
    if (F->hasAttr<clang::CUDADeviceAttr>()) {
      F->setInlineSpecified(true);
    }
  }
  return Result(D, true);
}

} // namespace cling
