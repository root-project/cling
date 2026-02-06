//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Bianca-Cristina Cristescu <bianca-cristina.cristescu@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file follows the same structure/logic (duplicate) as:
// clang/Interpreter/CodeCompletion.h
//
//===----------------------------------------------------------------------===//

#ifndef CLING_CODE_COMPLETE_CONSUMER
#define CLING_CODE_COMPLETE_CONSUMER

#include "clang/Sema/CodeCompleteConsumer.h"

using namespace clang;

namespace clang {
  class CompilerInstance;
}

namespace cling {
  struct ClingCodeCompleter {
    ClingCodeCompleter() = default;
    std::string Prefix;

    /// \param[in] InterpCI The compiler instance that is used to trigger code
    ///                     completion.
    /// \param[in] Content The string where code completion is triggered.
    /// \param[in] Line The line number of the code completion point.
    /// \param[in] Col The column number of the code completion point.
    /// \param[in] ParentCI The running interpreter compiler instance that
    ///                     provides ASTContexts.
    /// \param[out] CCResults The completion results.
    void codeComplete(clang::CompilerInstance* InterpCI,
                      llvm::StringRef Content, unsigned Line, unsigned Col,
                      clang::CompilerInstance* ParentCI,
                      std::vector<std::string>& CCResults);
  };
} // namespace cling

#endif
