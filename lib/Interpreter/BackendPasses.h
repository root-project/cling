//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_BACKENDPASSES_H
#define CLING_BACKENDPASSES_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/StandardInstrumentations.h"

#include <array>
#include <memory>

namespace llvm {
  class Function;
  class LLVMContext;
  class Module;
  class TargetMachine;
}

namespace clang {
  class CodeGenOptions;
  class LangOptions;
  class TargetOptions;
}

namespace cling {
  class IncrementalJIT;

  ///\brief Runs passes on IR. Remove once we can migrate from ModuleBuilder to
  /// what's in clang's CodeGen/BackendUtil.
  class BackendPasses {
    llvm::TargetMachine& m_TM;
    IncrementalJIT &m_JIT;
    const clang::CodeGenOptions &m_CGOpts;

    void CreatePasses(int OptLevel, llvm::ModulePassManager& MPM,
                      llvm::LoopAnalysisManager& LAM,
                      llvm::FunctionAnalysisManager& FAM,
                      llvm::CGSCCAnalysisManager& CGAM,
                      llvm::ModuleAnalysisManager& MAM,
                      llvm::PassInstrumentationCallbacks& PIC,
                      llvm::StandardInstrumentations& SI);

  public:
    BackendPasses(const clang::CodeGenOptions &CGOpts, IncrementalJIT &JIT,
                  llvm::TargetMachine& TM);
    ~BackendPasses();

    void runOnModule(llvm::Module& M, int OptLevel);
  };
}

#endif // CLING_BACKENDPASSES_H
