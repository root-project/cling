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
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/PassManager.h"

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CodeGenOptions.h"

using namespace cling;
using namespace clang;
using namespace llvm;

namespace {

  class InlinerKeepDeadFunc: public Inliner {
    Inliner* m_Inliner; // the actual inliner
    static char ID; // Pass identification, replacement for typeid
  public:
    InlinerKeepDeadFunc():
    Inliner(ID), m_Inliner(0) { }
    InlinerKeepDeadFunc(Pass* I):
    Inliner(ID), m_Inliner((Inliner*)I) { }

    using llvm::Pass::doInitialization;
    bool doInitialization(CallGraph &CG) override {
      // Forward out Resolver now that we are registered.
      if (!m_Inliner->getResolver())
        m_Inliner->setResolver(getResolver());
      return m_Inliner->doInitialization(CG); // no Module modification
    }

    InlineCost getInlineCost(CallSite CS) override {
      return m_Inliner->getInlineCost(CS);
    }
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      m_Inliner->getAnalysisUsage(AU);
    }
    bool runOnSCC(CallGraphSCC &SCC) override {
      return m_Inliner->runOnSCC(SCC);
    }

    using llvm::Pass::doFinalization;
    // No-op: we need to keep the inlined functions for later use.
    bool doFinalization(CallGraph& /*CG*/) override {
      // Module is unchanged
      return false;
    }
  };
} // end anonymous namespace

// Pass registration. Luckily all known inliners depend on the same set
// of passes.
char InlinerKeepDeadFunc::ID = 0;


BackendPasses::BackendPasses(const CodeGenOptions &CGOpts,
                             const TargetOptions &TOpts,
                             const LangOptions &LOpts):
  m_CodeGenOptsVerifyModule(CGOpts.VerifyModule)
{
  CreatePasses(CGOpts, TOpts, LOpts);
}


BackendPasses::~BackendPasses() {
  delete m_PMBuilder->Inliner;
}

void BackendPasses::CreatePasses(const CodeGenOptions &CGOpts,
                                 const TargetOptions &TOpts,
                                 const LangOptions &LOpts)
{
  // From BackEndUtil's clang::EmitAssemblyHelper::CreatePasses().

  unsigned OptLevel = CGOpts.OptimizationLevel;
  CodeGenOptions::InliningMethod Inlining = CGOpts.getInlining();

  // Handle disabling of LLVM optimization, where we want to preserve the
  // internal module before any optimization.
  if (CGOpts.DisableLLVMOpts) {
    OptLevel = 0;
    // Always keep at least ForceInline - NoInlining is deadly for libc++.
    // Inlining = CGOpts.NoInlining;
  }

  m_PMBuilder.reset(new PassManagerBuilder());
  m_PMBuilder->OptLevel = OptLevel;
  m_PMBuilder->SizeLevel = CGOpts.OptimizeSize;
  m_PMBuilder->BBVectorize = CGOpts.VectorizeBB;
  m_PMBuilder->SLPVectorize = CGOpts.VectorizeSLP;
  m_PMBuilder->LoopVectorize = CGOpts.VectorizeLoop;

  m_PMBuilder->DisableTailCalls = CGOpts.DisableTailCalls;
  m_PMBuilder->DisableUnitAtATime = !CGOpts.UnitAtATime;
  m_PMBuilder->DisableUnrollLoops = !CGOpts.UnrollLoops;
  m_PMBuilder->MergeFunctions = CGOpts.MergeFunctions;
  m_PMBuilder->RerollLoops = CGOpts.RerollLoops;


  switch (Inlining) {
    case CodeGenOptions::NoInlining: {
      assert(0 && "libc++ requires at least OnlyAlwaysInlining!");
      break;
    }
    case CodeGenOptions::NormalInlining: {
      m_PMBuilder->Inliner =
        new InlinerKeepDeadFunc(createFunctionInliningPass(OptLevel,
                                                          CGOpts.OptimizeSize));
      break;
    }
    case CodeGenOptions::OnlyAlwaysInlining:
      // Respect always_inline.
      if (OptLevel == 0)
        // Do not insert lifetime intrinsics at -O0.
        m_PMBuilder->Inliner
          = new InlinerKeepDeadFunc(createAlwaysInlinerPass(false));
      else
        m_PMBuilder->Inliner
          = new InlinerKeepDeadFunc(createAlwaysInlinerPass());
      break;
  }

  // Set up the per-module pass manager.
  m_MPM.reset(new PassManager());
  m_MPM->add(new DataLayoutPass());
  //m_MPM->add(createTargetTransformInfoWrapperPass(getTargetIRAnalysis()));
  //if (!CGOpts.RewriteMapFiles.empty())
  //  addSymbolRewriterPass(CGOpts, m_MPM);
  if (CGOpts.VerifyModule)
    m_MPM->add(createDebugInfoVerifierPass());

  m_PMBuilder->populateModulePassManager(*m_MPM);
}

void BackendPasses::runOnModule(Module& M) {

  // Set up the per-function pass manager.
  FunctionPassManager FPM(&M);
  FPM.add(new DataLayoutPass());
  //FPM.add(createTargetTransformInfoWrapperPass(getTargetIRAnalysis()));
  if (m_CodeGenOptsVerifyModule)
      FPM.add(createVerifierPass());
  m_PMBuilder->populateFunctionPassManager(FPM);

  // Run the per-function passes on the module.
  FPM.doInitialization();
  for (auto&& I: M.functions())
    if (!I.isDeclaration())
      FPM.run(I);
  FPM.doFinalization();

  m_MPM->run(M);
}
