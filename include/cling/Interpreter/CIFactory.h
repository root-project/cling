//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_CIFACTORY_H
#define CLING_CIFACTORY_H

#include "clang/Frontend/CompilerInstance.h"

#include "llvm/ADT/StringRef.h"

namespace llvm {
  class LLVMContext;
  class MemoryBuffer;
}

namespace clang {
  class DiagnosticsEngine;
  class ASTConsumer;
}

namespace cling {
  class InvocationOptions;

  namespace CIFactory {
    typedef std::unique_ptr<llvm::MemoryBuffer> MemBufPtr_t;

    // TODO: Add overload that takes file not MemoryBuffer

    clang::CompilerInstance*
    createCI(llvm::StringRef Code, const InvocationOptions& Opts,
             const char* LLVMDir, std::unique_ptr<clang::ASTConsumer> consumer);

    clang::CompilerInstance*
    createCI(MemBufPtr_t Buffer, int Argc, const char* const* Argv,
             const char* LLVMDir, std::unique_ptr<clang::ASTConsumer> consumer,
             bool OnlyLex = false);
  } // namespace CIFactory
} // namespace cling
#endif // CLING_CIFACTORY_H
