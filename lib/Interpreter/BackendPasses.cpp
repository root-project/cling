//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "BackendPasses.h"

#include "IncrementalJIT.h"

#include "cling/Utils/Platform.h"

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

      // It doesn't make sense to keep unnamed constants, we wouldn't know how
      // to reference them anyway.
      if (!GV.hasName())
        return false;

      if (GV.getName().startswith(".str"))
        return false;

      llvm::GlobalValue::LinkageTypes LT = GV.getLinkage();
      if (!GV.isDiscardableIfUnused(LT))
        return false;

      if (LT == llvm::GlobalValue::InternalLinkage) {
        // We want to keep this GlobalValue around, but have to tell the JIT
        // linker that it should not error on duplicate symbols.
        // FIXME: Ideally the frontend would never emit duplicate symbols and
        // we could just use the old version of saying:
        // GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
        GV.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
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
  class PreventLocalOptPass: public ModulePass {
    static char ID;

    bool runOnGlobal(GlobalValue& GV) {
      if (!GV.isDeclaration())
        return false; // no change.

      // GV is a declaration with no definition. Make sure to prevent any
      // optimization that tries to take advantage of the actual definition
      // being "local" because we have no influence on the memory layout of
      // data sections and how "close" they are to the code.

      bool changed = false;

      if (GV.hasLocalLinkage()) {
        GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
        changed = true;
      }

      if (!GV.hasDefaultVisibility()) {
        GV.setVisibility(llvm::GlobalValue::DefaultVisibility);
        changed = true;
      }

      // Set DSO locality last because setLinkage() and setVisibility() check
      // isImplicitDSOLocal().
      if (GV.isDSOLocal()) {
        GV.setDSOLocal(false);
        changed = true;
      }

      return changed;
    }

  public:
    PreventLocalOptPass() : ModulePass(ID) {}

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

char PreventLocalOptPass::ID = 0;

namespace {
  class WeakTypeinfoVTablePass: public ModulePass {
    static char ID;

    bool runOnGlobalVariable(GlobalVariable& GV) {
      // Only need to consider symbols with external linkage because only
      // these could be reported as duplicate.
      if (GV.getLinkage() != llvm::GlobalValue::ExternalLinkage)
        return false;

      if (GV.getName().startswith("_ZT")) {
        // Currently, if Cling sees the "key function" of a virtual class, it
        // emits typeinfo and vtable variables in every transaction llvm::Module
        // that reference them. Turn them into weak linkage to avoid duplicate
        // symbol errors from the JIT linker.
        // FIXME: This is a hack, we should teach the frontend to emit these
        // only once, or mark all duplicates as available_externally (if that
        // improves performance due to optimizations).
        GV.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
        return true; // a change!
      }

      return false;
    }

  public:
    WeakTypeinfoVTablePass() : ModulePass(ID) {}

    bool runOnModule(Module &M) override {
      bool ret = false;
      for (auto &&GV : M.globals())
        ret |= runOnGlobalVariable(GV);
      return ret;
    }
  };
}

char WeakTypeinfoVTablePass::ID = 0;

namespace {

  // Add a suffix to the CUDA module ctor/dtor, CUDA specific functions and
  // variables to generate a unique name. This is necessary for lazy
  // compilation. Without suffix, cling cannot distinguish ctor/dtor, register
  // function and and ptx code string of subsequent modules.
  class UniqueCUDAStructorName : public ModulePass {
    static char ID;

    // append a suffix to a symbol to make it unique
    // the suffix is "_cling_module_<module number>"
    llvm::SmallString<128> add_module_suffix(const StringRef SymbolName,
                                             const StringRef ModuleName) {
      llvm::SmallString<128> NewFunctionName;
      NewFunctionName.append(SymbolName);
      NewFunctionName.append("_");
      NewFunctionName.append(ModuleName);

      for (size_t i = 0; i < NewFunctionName.size(); ++i) {
        // Replace everything that is not [a-zA-Z0-9._] with a _. This set
        // happens to be the set of C preprocessing numbers.
        if (!isPreprocessingNumberBody(NewFunctionName[i]))
          NewFunctionName[i] = '_';
      }

      return NewFunctionName;
    }

    // make CUDA specific variables unique
    bool runOnGlobal(GlobalValue& GV, const StringRef ModuleName) {
      if (GV.isDeclaration())
        return false; // no change.

      if (!GV.hasName())
        return false;

      if (GV.getName().equals("__cuda_fatbin_wrapper") ||
          GV.getName().equals("__cuda_gpubin_handle")) {
        GV.setName(add_module_suffix(GV.getName(), ModuleName));
        return true;
      }

      return false;
    }

    // make CUDA specific functions unique
    bool runOnFunction(Function& F, const StringRef ModuleName) {
      if (F.hasName() && (F.getName().equals("__cuda_module_ctor") ||
                          F.getName().equals("__cuda_module_dtor") ||
                          F.getName().equals("__cuda_register_globals"))) {
        F.setName(add_module_suffix(F.getName(), ModuleName));
        return true;
      }

      return false;
    }

  public:
    UniqueCUDAStructorName() : ModulePass(ID) {}

    bool runOnModule(Module& M) override {
      bool ret = false;
      const StringRef ModuleName = M.getName();
      for (auto&& F : M)
        ret |= runOnFunction(F, ModuleName);
      for (auto&& G : M.globals())
        ret |= runOnGlobal(G, ModuleName);
      return ret;
    }
  };
} // namespace

char UniqueCUDAStructorName::ID = 0;


namespace {

  // Replace definitions of weak symbols for which symbols already exist by
  // declarations. This reduces the amount of emitted symbols.
  class ReuseExistingWeakSymbols : public ModulePass {
    static char ID;
    cling::IncrementalJIT &m_JIT;

    bool shouldRemoveGlobalDefinition(GlobalValue& GV) {
      // Existing *weak* symbols can be re-used thanks to ODR.
      llvm::GlobalValue::LinkageTypes LT = GV.getLinkage();
      if (!GV.isDiscardableIfUnused(LT) || !GV.isWeakForLinker(LT))
        return false;

      // Find the symbol as existing, previously compiled symbol in the JIT...
      if (m_JIT.doesSymbolAlreadyExist(GV.getName()))
        return true;

      // ...or in shared libraries (without auto-loading).
      std::string Name = GV.getName().str();
#if !defined(_WIN32)
        return llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(Name);
#else
        return platform::DLSym(Name);
#endif
    }

    bool runOnVar(GlobalVariable& GV) {
#if !defined(_WIN32)
      // Heuristically, Windows cannot handle cross-library variables; they
      // must be library-local.

      if (GV.isDeclaration())
        return false; // no change.
      if (shouldRemoveGlobalDefinition(GV)) {
        GV.setInitializer(nullptr); // make this a declaration
        return true; // a change!
      }
#endif
      return false; // no change.
    }

    bool runOnFunc(Function& Func) {
      if (Func.isDeclaration())
        return false; // no change.
#ifndef _WIN32
      // MSVC's stdlib gets symbol issues; i.e. apparently: JIT all or none.
      if (Func.getInstructionCount() < 50) {
        // This is a small function. Keep its definition to retain it for
        // inlining: the cost for JITting it is small, and the likelihood
        // that the call will be inlined is high.
        return false;
      }
#endif
      if (shouldRemoveGlobalDefinition(Func)) {
        Func.deleteBody(); // make this a declaration
        return true; // a change!
      }
      return false; // no change.
    }

  public:
    ReuseExistingWeakSymbols(IncrementalJIT &JIT) :
      ModulePass(ID), m_JIT(JIT) {}

    bool runOnModule(Module &M) override {
      bool ret = false;
      // FIXME: use SymbolLookupSet, rather than looking up symbol by symbol.
      for (auto &&F: M)
        ret |= runOnFunc(F);
      for (auto &&G: M.globals())
        ret |= runOnVar(G);
      return ret;
    }
  };
}

char ReuseExistingWeakSymbols::ID = 0;


BackendPasses::BackendPasses(const clang::CodeGenOptions &CGOpts,
                             IncrementalJIT &JIT, llvm::TargetMachine& TM):
   m_TM(TM),
   m_JIT(JIT),
   m_CGOpts(CGOpts)
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
  m_MPM[OptLevel]->add(new PreventLocalOptPass());
  m_MPM[OptLevel]->add(new WeakTypeinfoVTablePass());
  m_MPM[OptLevel]->add(new ReuseExistingWeakSymbols(m_JIT));

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
