//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "BackendPasses.h"

#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"

//#include "clang/Basic/LangOptions.h"
//#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/CodeGenOptions.h"

using namespace cling;
using namespace clang;
using namespace llvm;
using namespace llvm::legacy;

namespace {
  class KeepLocalGVPass: public ModulePass {
    static char ID;

    bool runOnGlobal(GlobalValue& GV) {
      if (GV.isDeclaration())
        return false; // no change.

      // GV is a definition.

      llvm::GlobalValue::LinkageTypes LT = GV.getLinkage();
      if (!GV.isDiscardableIfUnused(LT))
        return false;

      if (LT == llvm::GlobalValue::InternalLinkage
          || LT == llvm::GlobalValue::PrivateLinkage) {
        GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
        return true; // a change!
      }
      return false;
    }

  public:
    KeepLocalGVPass() : ModulePass(ID) {}

    bool runOnModule(Module &M) override {
      bool ret = false;
      for (auto &&F: M)
        ret |= runOnGlobal(F);
      for (auto &&G: M.globals())
        ret |= runOnGlobal(G);
      return ret;
    }
  };
}

char KeepLocalGVPass::ID = 0;

namespace {

  // Add a suffix to the CUDA module ctor/dtor to generate a unique name.
  // This is necessary for lazy compilation. Without suffix, cling cannot
  // distinguish ctor/dtor of subsequent modules.
  class UniqueCUDAStructorName : public ModulePass {
    static char ID;

    bool runOnFunction(Function& F, const StringRef ModuleName){
      if(F.hasName() && (F.getName() == "__cuda_module_ctor"
          || F.getName() == "__cuda_module_dtor") ){
        llvm::SmallString<128> NewFunctionName;
        NewFunctionName.append(F.getName());
        NewFunctionName.append("_");
        NewFunctionName.append(ModuleName);

        for (size_t i = 0; i < NewFunctionName.size(); ++i) {
          // Replace everything that is not [a-zA-Z0-9._] with a _. This set
          // happens to be the set of C preprocessing numbers.
          if (!isPreprocessingNumberBody(NewFunctionName[i]))
            NewFunctionName[i] = '_';
        }

        F.setName(NewFunctionName);

        return true;
      }

      return false;
    }

  public:
    UniqueCUDAStructorName() : ModulePass(ID) {}

    bool runOnModule(Module &M) override {
      bool ret = false;
      const StringRef ModuleName = M.getName();
      for (auto &&F: M)
        ret |= runOnFunction(F, ModuleName);
      return ret;
    }
  };
}

char UniqueCUDAStructorName::ID = 0;

BackendPasses::BackendPasses(const clang::CodeGenOptions &CGOpts,
                             const clang::TargetOptions & /*TOpts*/,
                             const clang::LangOptions & /*LOpts*/,
                             llvm::TargetMachine& TM):
   m_TM(TM),
   m_CGOpts(CGOpts) //,
   //m_TOpts(TOpts),
   //m_LOpts(LOpts)
{}


BackendPasses::~BackendPasses() {
  //delete m_PMBuilder->Inliner;
}

void BackendPasses::CreatePasses(llvm::Module& M, int OptLevel)
{
  // From BackEndUtil's clang::EmitAssemblyHelper::CreatePasses().

#if 0
  CodeGenOptions::InliningMethod Inlining = m_CGOpts.getInlining();
  CodeGenOptions& CGOpts_ = const_cast<CodeGenOptions&>(m_CGOpts);
  // DON'T: we will not find our symbols...
  //CGOpts_.CXXCtorDtorAliases = 1;

  // Default clang -O2 on Linux 64bit also has the following, but see
  // CIFactory.cpp.
  CGOpts_.DisableFPElim = 0;
  CGOpts_.DiscardValueNames = 1;
  CGOpts_.OmitLeafFramePointer = 1;
  CGOpts_.OptimizationLevel = 2;
  CGOpts_.RelaxAll = 0;
  CGOpts_.UnrollLoops = 1;
  CGOpts_.VectorizeLoop = 1;
  CGOpts_.VectorizeSLP = 1;
#endif

#if 0 // def __GNUC__
  // Better inlining is pending https://bugs.llvm.org//show_bug.cgi?id=19668
  // and its consequence https://sft.its.cern.ch/jira/browse/ROOT-7111
  // shown e.g. by roottest/cling/stl/map/badstringMap
  if (Inlining > CodeGenOptions::NormalInlining)
    Inlining = CodeGenOptions::NormalInlining;
#endif

  // Handle disabling of LLVM optimization, where we want to preserve the
  // internal module before any optimization.
  if (m_CGOpts.DisableLLVMPasses) {
    OptLevel = 0;
    // Always keep at least ForceInline - NoInlining is deadly for libc++.
    // Inlining = CGOpts.NoInlining;
  }

  llvm::PassManagerBuilder PMBuilder;
  PMBuilder.OptLevel = OptLevel;
  PMBuilder.SizeLevel = m_CGOpts.OptimizeSize;
  PMBuilder.SLPVectorize = OptLevel > 1 ? 1 : 0; // m_CGOpts.VectorizeSLP
  PMBuilder.LoopVectorize = OptLevel > 1 ? 1 : 0; // m_CGOpts.VectorizeLoop

  PMBuilder.DisableTailCalls = m_CGOpts.DisableTailCalls;
  PMBuilder.DisableUnrollLoops = !m_CGOpts.UnrollLoops;
  PMBuilder.MergeFunctions = m_CGOpts.MergeFunctions;
  PMBuilder.RerollLoops = m_CGOpts.RerollLoops;

  PMBuilder.LibraryInfo = new TargetLibraryInfoImpl(m_TM.getTargetTriple());

  // At O0 and O1 we only run the always inliner which is more efficient. At
  // higher optimization levels we run the normal inliner.
  // See also call to `CGOpts.setInlining()` in CIFactory!
  if (PMBuilder.OptLevel <= 1) {
    bool InsertLifetimeIntrinsics = PMBuilder.OptLevel != 0;
    PMBuilder.Inliner = createAlwaysInlinerLegacyPass(InsertLifetimeIntrinsics);
  } else {
    PMBuilder.Inliner = createFunctionInliningPass(OptLevel,
                                                   PMBuilder.SizeLevel,
            (!m_CGOpts.SampleProfileFile.empty() && m_CGOpts.PrepareForThinLTO));
  }

  // Set up the per-module pass manager.
  m_MPM[OptLevel].reset(new legacy::PassManager());

  m_MPM[OptLevel]->add(new KeepLocalGVPass());
  // The function __cuda_module_ctor and __cuda_module_dtor will just generated,
  // if a CUDA fatbinary file exist. Without file path there is no need for the
  // function pass.
  if(!m_CGOpts.CudaGpuBinaryFileName.empty())
    m_MPM[OptLevel]->add(new UniqueCUDAStructorName());
  m_MPM[OptLevel]->add(createTargetTransformInfoWrapperPass(
                                                   m_TM.getTargetIRAnalysis()));

  m_TM.adjustPassManager(PMBuilder);

  PMBuilder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                         [&](const PassManagerBuilder &,
                             legacy::PassManagerBase &PM) {
                              PM.add(createAddDiscriminatorsPass());
                            });

  //if (!CGOpts.RewriteMapFiles.empty())
  //  addSymbolRewriterPass(CGOpts, m_MPM);

  PMBuilder.populateModulePassManager(*m_MPM[OptLevel]);

  m_FPM[OptLevel].reset(new legacy::FunctionPassManager(&M));
  m_FPM[OptLevel]->add(createTargetTransformInfoWrapperPass(
                                                   m_TM.getTargetIRAnalysis()));
  if (m_CGOpts.VerifyModule)
      m_FPM[OptLevel]->add(createVerifierPass());
  PMBuilder.populateFunctionPassManager(*m_FPM[OptLevel]);
}

void BackendPasses::runOnModule(Module& M, int OptLevel) {

  if (OptLevel < 0)
    OptLevel = 0;
  if (OptLevel > 3)
    OptLevel = 3;

  if (!m_MPM[OptLevel])
    CreatePasses(M, OptLevel);

  static constexpr std::array<llvm::CodeGenOpt::Level, 4> CGOptLevel {{
    llvm::CodeGenOpt::None,
    llvm::CodeGenOpt::Less,
    llvm::CodeGenOpt::Default,
    llvm::CodeGenOpt::Aggressive
  }};
  // TM's OptLevel is used to build orc::SimpleCompiler passes for every Module.
  m_TM.setOptLevel(CGOptLevel[OptLevel]);

  // Run the per-function passes on the module.
  m_FPM[OptLevel]->doInitialization();
  for (auto&& I: M.functions())
    if (!I.isDeclaration())
      m_FPM[OptLevel]->run(I);
  m_FPM[OptLevel]->doFinalization();

  m_MPM[OptLevel]->run(M);
}
