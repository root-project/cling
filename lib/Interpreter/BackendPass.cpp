//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "BackendPass.h"

#include "llvm/Analysis/InlineCost.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/PassManager.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Scalar.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "clang/Frontend/FrontendDiagnostic.h"

using namespace llvm;
using namespace clang;

namespace {

  class InlinerKeepDeadFunc: public CallGraphSCCPass {
    Inliner* m_Inliner; // the actual inliner
    static char ID; // Pass identification, replacement for typeid
  public:
    InlinerKeepDeadFunc():
      CallGraphSCCPass(ID), m_Inliner(0) { }
    InlinerKeepDeadFunc(Pass* I):
      CallGraphSCCPass(ID), m_Inliner((Inliner*)I) { }

    bool doInitialization(CallGraph &CG) {
      // Forward out Resolver now that we are registered.
      if (!m_Inliner->getResolver())
        m_Inliner->setResolver(getResolver());
      return m_Inliner->doInitialization(CG); // no Module modification
    }

    InlineCost getInlineCost(CallSite CS) {
      return m_Inliner->getInlineCost(CS);
    }
    void getAnalysisUsage(AnalysisUsage &AU) const {
      m_Inliner->getAnalysisUsage(AU);
    }
    bool runOnSCC(CallGraphSCC &SCC) {
      return m_Inliner->runOnSCC(SCC);
    }

    using llvm::Pass::doFinalization;
    // No-op: we need to keep the inlined functions for later use.
    bool doFinalization(CallGraph& /*CG*/) {
      // Module is unchanged
      return false;
    }
  };
} // end anonymous namespace

// Pass registration. Luckily all known inliners depend on the same set
// of passes.
char InlinerKeepDeadFunc::ID = 0;

namespace {
  class PassManagerBuilderWithOpts: public PassManagerBuilder {
  public:
    const clang::LangOptions& m_LangOpts;
    const clang::CodeGenOptions& m_CodeGenOpts;
    PassManagerBuilderWithOpts(const clang::LangOptions& LangOpts,
                              const clang::CodeGenOptions& CodeGenOpts):
      m_LangOpts(LangOpts), m_CodeGenOpts(CodeGenOpts) {}

    const clang::LangOptions& getLangOpts() const { return m_LangOpts; }
    const clang::CodeGenOptions& getCGOpts() const { return m_CodeGenOpts; }
  };

static void
addObjCARCAPElimPass(const PassManagerBuilder &Builder, PassManagerBase &PM) {
  if (Builder.OptLevel > 0)
    PM.add(createObjCARCAPElimPass());
}

static void
addObjCARCExpandPass(const PassManagerBuilder &Builder, PassManagerBase &PM) {
  if (Builder.OptLevel > 0)
    PM.add(createObjCARCExpandPass());
}

static void
addObjCARCOptPass(const PassManagerBuilder &Builder, PassManagerBase &PM) {
  if (Builder.OptLevel > 0)
    PM.add(createObjCARCOptPass());
}

#if 0
static void addSampleProfileLoaderPass(const PassManagerBuilder &Builder,
                                       PassManagerBase &PM) {
  const PassManagerBuilderWithOpts &BuilderWrapper =
      static_cast<const PassManagerBuilderWithOpts &>(Builder);
  const CodeGenOptions &CGOpts = BuilderWrapper.getCGOpts();
  PM.add(createSampleProfileLoaderPass(CGOpts.SampleProfileFile));
}
#endif

static void addBoundsCheckingPass(const PassManagerBuilder &Builder,
                                    PassManagerBase &PM) {
  PM.add(createBoundsCheckingPass());
}

static void addAddressSanitizerPasses(const PassManagerBuilder &Builder,
                                      PassManagerBase &PM) {
  const PassManagerBuilderWithOpts &BuilderWrapper =
      static_cast<const PassManagerBuilderWithOpts&>(Builder);
  const CodeGenOptions &CGOpts = BuilderWrapper.getCGOpts();
  const LangOptions &LangOpts = BuilderWrapper.getLangOpts();
  PM.add(createAddressSanitizerFunctionPass(
      LangOpts.Sanitize.InitOrder,
      LangOpts.Sanitize.UseAfterReturn,
      LangOpts.Sanitize.UseAfterScope,
      CGOpts.SanitizerBlacklistFile));
  PM.add(createAddressSanitizerModulePass(
      LangOpts.Sanitize.InitOrder,
      CGOpts.SanitizerBlacklistFile));
}

static void addMemorySanitizerPass(const PassManagerBuilder &Builder,
                                   PassManagerBase &PM) {
  const PassManagerBuilderWithOpts &BuilderWrapper =
      static_cast<const PassManagerBuilderWithOpts&>(Builder);
  const CodeGenOptions &CGOpts = BuilderWrapper.getCGOpts();
  PM.add(createMemorySanitizerPass(CGOpts.SanitizeMemoryTrackOrigins,
                                   CGOpts.SanitizerBlacklistFile));

  // MemorySanitizer inserts complex instrumentation that mostly follows
  // the logic of the original code, but operates on "shadow" values.
  // It can benefit from re-running some general purpose optimization passes.
  if (Builder.OptLevel > 0) {
    PM.add(createEarlyCSEPass());
    PM.add(createReassociatePass());
    PM.add(createLICMPass());
    PM.add(createGVNPass());
    PM.add(createInstructionCombiningPass());
    PM.add(createDeadStoreEliminationPass());
  }
}

static void addThreadSanitizerPass(const PassManagerBuilder &Builder,
                                   PassManagerBase &PM) {
  const PassManagerBuilderWithOpts &BuilderWrapper =
      static_cast<const PassManagerBuilderWithOpts&>(Builder);
  const CodeGenOptions &CGOpts = BuilderWrapper.getCGOpts();
  PM.add(createThreadSanitizerPass(CGOpts.SanitizerBlacklistFile));
}

static void addDataFlowSanitizerPass(const PassManagerBuilder &Builder,
                                     PassManagerBase &PM) {
  const PassManagerBuilderWithOpts &BuilderWrapper =
      static_cast<const PassManagerBuilderWithOpts&>(Builder);
  const CodeGenOptions &CGOpts = BuilderWrapper.getCGOpts();
  PM.add(createDataFlowSanitizerPass(CGOpts.SanitizerBlacklistFile));
}
}

namespace cling {

  BackendPass::BackendPass(clang::Sema* S, llvm::Module* M,
                           clang::DiagnosticsEngine& Diags,
                           const clang::TargetOptions& TOpts,
                           const clang::LangOptions& LangOpts,
                           const clang::CodeGenOptions& CodeGenOpts):
    TransactionTransformer(S), m_Module(M) {
    m_TM.reset(CreateTargetMachine(Diags, TOpts, LangOpts, CodeGenOpts));
    if (m_TM)
      CreatePasses(LangOpts, CodeGenOpts);
  }

  llvm::FunctionPassManager *BackendPass::getPerFunctionPasses() {
    if (!m_PerFunctionPasses) {
      m_PerFunctionPasses.reset(new FunctionPassManager(m_Module));
      m_PerFunctionPasses->add(new DataLayoutPass(m_Module));
      m_TM->addAnalysisPasses(*m_PerFunctionPasses);
    }
    return m_PerFunctionPasses.get();
  }

  llvm::PassManager *BackendPass::getPerModulePasses() {
    if (!m_PerModulePasses) {
      m_PerModulePasses.reset(new PassManager());
      m_PerModulePasses->add(new DataLayoutPass(m_Module));
      m_TM->addAnalysisPasses(*m_PerModulePasses);
    }
    return m_PerModulePasses.get();
  }

  llvm::TargetMachine*
  BackendPass::CreateTargetMachine(clang::DiagnosticsEngine& Diags,
                                   const clang::TargetOptions& TargetOpts,
                                   const clang::LangOptions& LangOpts,
                                   const clang::CodeGenOptions& CodeGenOpts) {
    // Create the TargetMachine for generating code.
    // FIXME: Expose these capabilities via actual APIs!!!! Aside from just
    // being gross, this is also totally broken if we ever care about
    // concurrency.

    TargetMachine::setAsmVerbosityDefault(CodeGenOpts.AsmVerbose);

    TargetMachine::setFunctionSections(CodeGenOpts.FunctionSections);
    TargetMachine::setDataSections    (CodeGenOpts.DataSections);

    // FIXME: Parse this earlier.
    llvm::CodeModel::Model CM;
    if (CodeGenOpts.CodeModel == "small") {
      CM = llvm::CodeModel::Small;
    } else if (CodeGenOpts.CodeModel == "kernel") {
      CM = llvm::CodeModel::Kernel;
    } else if (CodeGenOpts.CodeModel == "medium") {
      CM = llvm::CodeModel::Medium;
    } else if (CodeGenOpts.CodeModel == "large") {
      CM = llvm::CodeModel::Large;
    } else {
      assert(CodeGenOpts.CodeModel.empty() && "Invalid code model!");
      CM = llvm::CodeModel::Default;
    }

    SmallVector<const char *, 16> BackendArgs;
    BackendArgs.push_back("cling"); // Fake program name.
    if (!CodeGenOpts.DebugPass.empty()) {
      BackendArgs.push_back("-debug-pass");
      BackendArgs.push_back(CodeGenOpts.DebugPass.c_str());
    }
    if (!CodeGenOpts.LimitFloatPrecision.empty()) {
      BackendArgs.push_back("-limit-float-precision");
      BackendArgs.push_back(CodeGenOpts.LimitFloatPrecision.c_str());
    }
    if (llvm::TimePassesIsEnabled)
      BackendArgs.push_back("-time-passes");
    for (unsigned i = 0, e = CodeGenOpts.BackendOptions.size(); i != e; ++i)
      BackendArgs.push_back(CodeGenOpts.BackendOptions[i].c_str());
    if (CodeGenOpts.NoGlobalMerge)
      BackendArgs.push_back("-global-merge=false");
    BackendArgs.push_back(0);
    llvm::cl::ParseCommandLineOptions(BackendArgs.size() - 1,
                                      BackendArgs.data());

    std::string FeaturesStr;
    if (TargetOpts.Features.size()) {
      SubtargetFeatures Features;
      for (std::vector<std::string>::const_iterator
             it = TargetOpts.Features.begin(),
             ie = TargetOpts.Features.end(); it != ie; ++it)
        Features.AddFeature(*it);
      FeaturesStr = Features.getString();
    }

    llvm::Reloc::Model RM = llvm::Reloc::Default;
    if (CodeGenOpts.RelocationModel == "static") {
      RM = llvm::Reloc::Static;
    } else if (CodeGenOpts.RelocationModel == "pic") {
      RM = llvm::Reloc::PIC_;
    } else {
      assert(CodeGenOpts.RelocationModel == "dynamic-no-pic" &&
             "Invalid PIC model!");
      RM = llvm::Reloc::DynamicNoPIC;
    }

    CodeGenOpt::Level OptLevel = CodeGenOpt::Default;
    switch (CodeGenOpts.OptimizationLevel) {
    default: break;
    case 0: OptLevel = CodeGenOpt::None; break;
    case 3: OptLevel = CodeGenOpt::Aggressive; break;
    }

    llvm::TargetOptions Options;

    if (CodeGenOpts.DisableIntegratedAS)
      Options.DisableIntegratedAS = true;

    // Set frame pointer elimination mode.
    if (!CodeGenOpts.DisableFPElim) {
      Options.NoFramePointerElim = false;
    } else if (CodeGenOpts.OmitLeafFramePointer) {
      Options.NoFramePointerElim = false;
    } else {
      Options.NoFramePointerElim = true;
    }

    if (CodeGenOpts.UseInitArray)
      Options.UseInitArray = true;

    // Set float ABI type.
    if (CodeGenOpts.FloatABI == "soft" || CodeGenOpts.FloatABI == "softfp")
      Options.FloatABIType = llvm::FloatABI::Soft;
    else if (CodeGenOpts.FloatABI == "hard")
      Options.FloatABIType = llvm::FloatABI::Hard;
    else {
      assert(CodeGenOpts.FloatABI.empty() && "Invalid float abi!");
      Options.FloatABIType = llvm::FloatABI::Default;
    }

    // Set FP fusion mode.
    switch (CodeGenOpts.getFPContractMode()) {
    case CodeGenOptions::FPC_Off:
      Options.AllowFPOpFusion = llvm::FPOpFusion::Strict;
      break;
    case CodeGenOptions::FPC_On:
      Options.AllowFPOpFusion = llvm::FPOpFusion::Standard;
      break;
    case CodeGenOptions::FPC_Fast:
      Options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
      break;
    }

    Options.LessPreciseFPMADOption = CodeGenOpts.LessPreciseFPMAD;
    Options.NoInfsFPMath = CodeGenOpts.NoInfsFPMath;
    Options.NoNaNsFPMath = CodeGenOpts.NoNaNsFPMath;
    Options.NoZerosInBSS = CodeGenOpts.NoZeroInitializedInBSS;
    Options.UnsafeFPMath = CodeGenOpts.UnsafeFPMath;
    Options.UseSoftFloat = CodeGenOpts.SoftFloat;
    Options.StackAlignmentOverride = CodeGenOpts.StackAlignment;
    Options.DisableTailCalls = CodeGenOpts.DisableTailCalls;
    Options.TrapFuncName = CodeGenOpts.TrapFuncName;
    Options.PositionIndependentExecutable = LangOpts.PIELevel != 0;
    Options.EnableSegmentedStacks = CodeGenOpts.EnableSegmentedStacks;

    Triple TheTriple;
    TheTriple.setTriple(sys::getProcessTriple());
    std::string Error;
    const Target* TheTarget
      = TargetRegistry::lookupTarget(TheTriple.getTriple(), Error);
    if (!TheTarget) {
      Diags.Report(diag::err_fe_unable_to_create_target) << Error;
      return 0;
    }

    TargetMachine *TM = TheTarget->createTargetMachine(TheTriple.getTriple(),
                                                       TargetOpts.CPU,
                                                       FeaturesStr, Options,
                                                       RM, CM, OptLevel);

    if (CodeGenOpts.RelaxAll)
      TM->setMCRelaxAll(true);
    if (CodeGenOpts.SaveTempLabels)
      TM->setMCSaveTempLabels(true);
    if (CodeGenOpts.NoDwarf2CFIAsm)
      TM->setMCUseCFI(false);
    if (!CodeGenOpts.NoDwarfDirectoryAsm)
      TM->setMCUseDwarfDirectory(true);
    if (CodeGenOpts.NoExecStack)
      TM->setMCNoExecStack(true);

    return TM;
  }

  void BackendPass::CreatePasses(const clang::LangOptions& LangOpts,
                                 const clang::CodeGenOptions& CodeGenOpts) {
    // See clang/lib/CodeGen/BackendUtil.cpp EmitAssemblyHelper::CreatePasses()
    unsigned OptLevel = CodeGenOpts.OptimizationLevel;
    CodeGenOptions::InliningMethod Inlining = CodeGenOpts.getInlining();

    // Handle disabling of LLVM optimization, where we want to preserve the
    // internal module before any optimization.
    if (CodeGenOpts.DisableLLVMOpts) {
      OptLevel = 0;
      Inlining = CodeGenOpts.NoInlining;
    }

    PassManagerBuilderWithOpts PMBuilder(LangOpts, CodeGenOpts);
    PMBuilder.OptLevel = OptLevel;
    PMBuilder.SizeLevel = CodeGenOpts.OptimizeSize;
    PMBuilder.BBVectorize = CodeGenOpts.VectorizeBB;
    PMBuilder.SLPVectorize = CodeGenOpts.VectorizeSLP;
    PMBuilder.LoopVectorize = CodeGenOpts.VectorizeLoop;

    PMBuilder.DisableUnitAtATime = !CodeGenOpts.UnitAtATime;
    PMBuilder.DisableUnrollLoops = !CodeGenOpts.UnrollLoops;
    PMBuilder.RerollLoops = CodeGenOpts.RerollLoops;

#if 0
    if (!CodeGenOpts.SampleProfileFile.empty())
      PMBuilder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                             addSampleProfileLoaderPass);
#endif

    // In ObjC ARC mode, add the main ARC optimization passes.
    if (LangOpts.ObjCAutoRefCount) {
      PMBuilder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                             addObjCARCExpandPass);
      PMBuilder.addExtension(PassManagerBuilder::EP_ModuleOptimizerEarly,
                             addObjCARCAPElimPass);
      PMBuilder.addExtension(PassManagerBuilder::EP_ScalarOptimizerLate,
                             addObjCARCOptPass);
    }

    if (LangOpts.Sanitize.LocalBounds) {
      PMBuilder.addExtension(PassManagerBuilder::EP_ScalarOptimizerLate,
                             addBoundsCheckingPass);
      PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                             addBoundsCheckingPass);
    }

    if (LangOpts.Sanitize.Address) {
      PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                             addAddressSanitizerPasses);
      PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                             addAddressSanitizerPasses);
    }

    if (LangOpts.Sanitize.Memory) {
      PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                             addMemorySanitizerPass);
      PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                             addMemorySanitizerPass);
    }

    if (LangOpts.Sanitize.Thread) {
      PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                             addThreadSanitizerPass);
      PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                             addThreadSanitizerPass);
    }

    if (LangOpts.Sanitize.DataFlow) {
      PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                             addDataFlowSanitizerPass);
      PMBuilder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                             addDataFlowSanitizerPass);
    }

    // Figure out TargetLibraryInfo.
    Triple TargetTriple(m_Module->getTargetTriple());
    PMBuilder.LibraryInfo = new TargetLibraryInfo(TargetTriple);
    if (!CodeGenOpts.SimplifyLibCalls)
      PMBuilder.LibraryInfo->disableAllFunctions();

    switch (Inlining) {
    case CodeGenOptions::NoInlining: break;
    case CodeGenOptions::NormalInlining: {
      // FIXME: Derive these constants in a principled fashion.
      unsigned Threshold = 225;
      if (CodeGenOpts.OptimizeSize == 1)      // -Os
        Threshold = 75;
      else if (CodeGenOpts.OptimizeSize == 2) // -Oz
        Threshold = 25;
      else if (OptLevel > 2)
        Threshold = 275;
      // Creates a SimpleInliner that requests InsertLifetime.
      PMBuilder.Inliner
        = new InlinerKeepDeadFunc(createFunctionInliningPass(Threshold));
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

    // Set up the per-function pass manager.
    FunctionPassManager *FPM = getPerFunctionPasses();
    if (CodeGenOpts.VerifyModule)
      FPM->add(createVerifierPass());
    PMBuilder.populateFunctionPassManager(*FPM);

    // The Inliner is a module pass; register it.
    PMBuilder.populateModulePassManager(*getPerModulePasses());
  }

  // pin the vtable and OwningPtrs' dtors.
  BackendPass::~BackendPass() {}

  void BackendPass::Transform() {
    // FIXME: This should not revisit the whole module but only its
    // llvm::Functions created by the current transaction.
    for (auto& F: *m_Module) {
      if (!F.isDeclaration())
        m_PerFunctionPasses->run(F);
    }
    // Do not remove force_inline functions: we might need them for
    // inlining them into the next function calling them, and CodeGen will
    // not emit them anymore.
    //m_PerFunctionPasses->doFinalization();

    if (m_PerModulePasses) {
      m_PerModulePasses->run(*m_Module);
    }
  }
} // end namespace cling
