//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
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
}

namespace cling {
  class CIFactory {
  public:
    // TODO: Add overload that takes file not MemoryBuffer
    static clang::CompilerInstance* createCI(llvm::StringRef code,
                                             int argc,
                                             const char* const *argv,
                                             const char* llvmdir);

    static clang::CompilerInstance* createCI(llvm::MemoryBuffer* buffer,
                                             int argc,
                                             const char* const *argv,
                                             const char* llvmdir);
  private:
    //---------------------------------------------------------------------
    //! Constructor
    //---------------------------------------------------------------------
    CIFactory() {}
    ~CIFactory() {}
    static void SetClingCustomLangOpts(clang::LangOptions& Opts);
    static void SetClingTargetLangOpts(clang::LangOptions& Opts,
                                       const clang::TargetInfo& Target);
  };
} // namespace cling
#endif // CLING_CIFACTORY_H
