//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "ASTTransformer.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclGroup.h"

namespace cling {

  // pin the vtable here since there is no point to create dedicated to that
  // cpp file.
  ASTTransformer::~ASTTransformer() {}

  void ASTTransformer::Emit(clang::DeclGroupRef DGR) {
    m_Consumer->HandleTopLevelDecl(DGR);
  }

} // end namespace cling
