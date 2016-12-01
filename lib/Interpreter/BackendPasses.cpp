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
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"

//#include "clang/Basic/LangOptions.h"
//#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CodeGenOptions.h"

using namespace cling;
using namespace clang;
using namespace llvm;
using namespace llvm::legacy;

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


BackendPasses::~BackendPasses() {
  //delete m_PMBuilder->Inliner;
}

void BackendPasses::CreatePasses(llvm::Module& M)
{
  // From BackEndUtil's clang::EmitAssemblyHelper::CreatePasses().

  CodeGenOptions& CGOpts_ = const_cast<CodeGenOptions&>(m_CGOpts);
  // DON'T: we will not find our symbols...
  //CGOpts_.CXXCtorDtorAliases = 1;

  // Default clang -O2 on Linux 64bit also has:
#if 0
  CGOpts_.DisableFPElim = 0;
  CGOpts_.DiscardValueNames = 1;
  CGOpts_.OmitLeafFramePointer = 1;
  CGOpts_.OptimizationLevel = 2;
  CGOpts_.RelaxAll = 0;
  CGOpts_.UnrollLoops = 1;
  CGOpts_.VectorizeLoop = 1;
  CGOpts_.VectorizeSLP = 1;
#endif
  CGOpts_.setInlining(CodeGenOptions::NormalInlining);

  unsigned OptLevel = m_CGOpts.OptimizationLevel;

  // CodeGenOptions::InliningMethod Inlining = m_CGOpts.getInlining();
  CodeGenOptions::InliningMethod Inlining = CodeGenOptions::NormalInlining;

  // Handle disabling of LLVM optimization, where we want to preserve the
  // internal module before any optimization.
  if (m_CGOpts.DisableLLVMOpts) {
    OptLevel = 0;
    // Always keep at least ForceInline - NoInlining is deadly for libc++.
    // Inlining = CGOpts.NoInlining;
  }

  OptLevel = 0; // we need to keep even "unused" values - until we
  // feed incremental modules into the JIT.

  llvm::PassManagerBuilder PMBuilder;
  PMBuilder.OptLevel = OptLevel;
  PMBuilder.SizeLevel = m_CGOpts.OptimizeSize;
  PMBuilder.BBVectorize = m_CGOpts.VectorizeBB;
  PMBuilder.SLPVectorize = m_CGOpts.VectorizeSLP;
  PMBuilder.LoopVectorize = m_CGOpts.VectorizeLoop;

  PMBuilder.DisableTailCalls = m_CGOpts.DisableTailCalls;
  PMBuilder.DisableUnitAtATime = !m_CGOpts.UnitAtATime;
  PMBuilder.DisableUnrollLoops = !m_CGOpts.UnrollLoops;
  PMBuilder.MergeFunctions = m_CGOpts.MergeFunctions;
  PMBuilder.RerollLoops = m_CGOpts.RerollLoops;

  PMBuilder.LibraryInfo = new TargetLibraryInfoImpl(m_TM.getTargetTriple());


  switch (Inlining) {
  case CodeGenOptions::OnlyHintInlining: // fall-through:
    case CodeGenOptions::NoInlining: {
      assert(0 && "libc++ requires at least OnlyAlwaysInlining!");
      break;
    }
    case CodeGenOptions::NormalInlining: {
      PMBuilder.Inliner =
        new InlinerKeepDeadFunc(createFunctionInliningPass(OptLevel,
                                                        m_CGOpts.OptimizeSize));
      break;
    }
    case CodeGenOptions::OnlyAlwaysInlining:
      // Respect always_inline.
      if (OptLevel == 0)
        // Do not insert lifetime intrinsics at -O0.
        PMBuilder.Inliner
          = new InlinerKeepDeadFunc(createAlwaysInlinerPass(false));
      else
        PMBuilder.Inliner
          = new InlinerKeepDeadFunc(createAlwaysInlinerPass());
      break;
  }

  // Set up the per-module pass manager.
  m_MPM.reset(new legacy::PassManager());

  m_MPM->add(createTargetTransformInfoWrapperPass(m_TM.getTargetIRAnalysis()));

  // Add target-specific passes that need to run as early as possible.
  PMBuilder.addExtension(
                         PassManagerBuilder::EP_EarlyAsPossible,
                         [&](const PassManagerBuilder &,
                             legacy::PassManagerBase &PM) {
                           m_TM.addEarlyAsPossiblePasses(PM);
                         });

  PMBuilder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                         [&](const PassManagerBuilder &,
                             legacy::PassManagerBase &PM) {
                              PM.add(createAddDiscriminatorsPass());
                            });

  //if (!CGOpts.RewriteMapFiles.empty())
  //  addSymbolRewriterPass(CGOpts, m_MPM);

  PMBuilder.populateModulePassManager(*m_MPM);

  m_FPM.reset(new legacy::FunctionPassManager(&M));
  m_FPM->add(createTargetTransformInfoWrapperPass(m_TM.getTargetIRAnalysis()));
  if (m_CGOpts.VerifyModule)
      m_FPM->add(createVerifierPass());
  PMBuilder.populateFunctionPassManager(*m_FPM);
}

void BackendPasses::runOnModule(Module& M) {

  if (!m_MPM)
    CreatePasses(M);
  // Set up the per-function pass manager.

  // Run the per-function passes on the module.
  m_FPM->doInitialization();
  for (auto&& I: M.functions())
    if (!I.isDeclaration())
      m_FPM->run(I);
  m_FPM->doFinalization();

  m_MPM->run(M);
}
