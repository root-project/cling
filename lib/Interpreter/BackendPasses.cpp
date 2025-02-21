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
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"

//#include "clang/Basic/LangOptions.h"
//#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/CodeGenOptions.h"

#include <optional>

using namespace cling;
using namespace clang;
using namespace llvm;

namespace {
  class UniqueInitFunctionNamePass
      : public PassInfoMixin<UniqueInitFunctionNamePass> {
    // append a suffix to a symbol to make it unique
    // the suffix is "_cling_module_<module number>"
    llvm::SmallString<128> add_module_suffix(const StringRef OriginalName,
                                             const StringRef ModuleName) {
      llvm::SmallString<128> NewFunctionName;
      NewFunctionName.append(ModuleName);
      NewFunctionName.append("_");

      for (size_t i = 0; i < NewFunctionName.size(); ++i) {
        // Replace everything that is not [a-zA-Z0-9._] with a _. This set
        // happens to be the set of C preprocessing numbers.
        if (!isPreprocessingNumberBody(NewFunctionName[i]))
          NewFunctionName[i] = '_';
      }

      return NewFunctionName;
    }

    // make static initialization function names (__cxx_global_var_init) unique
    bool runOnFunction(Function& F, const StringRef ModuleName) {
      if (F.hasName() && F.getName().starts_with("__cxx_global_var_init")) {
        F.setName(add_module_suffix(F.getName(), ModuleName));
        return true;
      }

      return false;
    }

  public:
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      const StringRef ModuleName = M.getName();
      for (auto&& F : M)
        changed |= runOnFunction(F, ModuleName);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
} // namespace

namespace {
  class WorkAroundConstructorPriorityBugPass
      : public PassInfoMixin<WorkAroundConstructorPriorityBugPass> {
  public:
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      llvm::GlobalVariable* GlobalCtors = M.getNamedGlobal("llvm.global_ctors");
      if (!GlobalCtors)
        return PreservedAnalyses::all();

      auto* OldCtors = llvm::dyn_cast_or_null<llvm::ConstantArray>(
          GlobalCtors->getInitializer());
      if (!OldCtors)
        return PreservedAnalyses::all();

      // LLVM had a bug where constructors with the same priority would not be
      // stably sorted. This has been fixed upstream by
      // https://github.com/llvm/llvm-project/pull/95532, but to avoid relying
      // on a backport this pass works around the issue: The idea is that we
      // lower the default priority of concerned constructors to make them sort
      // correctly.
      static constexpr uint64_t DefaultPriority = 65535;

      unsigned NumCtors = OldCtors->getNumOperands();
      const uint64_t NewDefaultPriorityStart = DefaultPriority - NumCtors;
      uint64_t NewDefaultPriority = NewDefaultPriorityStart;

      llvm::SmallVector<Constant*> NewCtors;
      for (unsigned I = 0; I < NumCtors; I++) {
        auto* Ctor =
            llvm::dyn_cast<llvm::ConstantStruct>(OldCtors->getOperand(I));
        auto* PriorityC = llvm::cast<llvm::ConstantInt>(Ctor->getOperand(0));
        uint64_t Priority = PriorityC->getZExtValue();
        if (Priority >= NewDefaultPriorityStart && Priority < DefaultPriority) {
          llvm::errs() << "Found priority " << Priority
                       << ", not changing anything\n";
          return PreservedAnalyses::all();
        }

        if (Priority == DefaultPriority) {
          Priority = NewDefaultPriority;
          NewDefaultPriority++;
        }

        llvm::SmallVector<Constant*> NewCtorArgs;
        NewCtorArgs.push_back(
            llvm::ConstantInt::get(PriorityC->getIntegerType(), Priority));
        // Copy the function and data Constant, if present.
        NewCtorArgs.push_back(Ctor->getOperand(1));
        if (Ctor->getNumOperands() >= 3) {
          NewCtorArgs.push_back(Ctor->getOperand(2));
        }

        NewCtors.push_back(
            llvm::ConstantStruct::get(Ctor->getType(), NewCtorArgs));
      }

      GlobalCtors->setInitializer(
          llvm::ConstantArray::get(OldCtors->getType(), NewCtors));

      return PreservedAnalyses::none();
    }
  };
} // namespace

namespace {
  class KeepLocalGVPass : public PassInfoMixin<KeepLocalGVPass> {
    bool runOnGlobal(GlobalValue& GV) {
      if (GV.isDeclaration())
        return false; // no change.

      // GV is a definition.

      // It doesn't make sense to keep unnamed constants, we wouldn't know how
      // to reference them anyway.
      if (!GV.hasName())
        return false;

      if (GV.getName().starts_with(".str"))
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
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      for (auto &&F: M)
        changed |= runOnGlobal(F);
      for (auto &&G: M.globals())
        changed |= runOnGlobal(G);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

namespace {
  class PreventLocalOptPass : public PassInfoMixin<PreventLocalOptPass> {
    bool runOnGlobal(GlobalValue& GV) {
      bool changed = false;

      // Prevent any optimization that tries to take advantage of the actual
      // definition being "local" because we have no influence on the memory
      // layout of sections and how "close" they are.

      if (GV.hasLocalLinkage()) {
        if (GV.isDeclaration()) {
          // For declarations with no definition, we can simply adjust the
          // linkage.
          GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
          changed = true;
        } else {
          // FIXME: Not clear what would be the right linkage. We also cannot
          // continue because "GlobalValue with local linkage [...] must be
          // dso_local!"
          return false;
        }
      }

      if (!GV.hasDefaultVisibility()) {
        GV.setVisibility(llvm::GlobalValue::DefaultVisibility);
        changed = true;
      }

      // Set DSO locality last because setLinkage() and setVisibility() check
      // isImplicitDSOLocal() and then might call setDSOLocal(true).
      if (GV.isDSOLocal()) {
        GV.setDSOLocal(false);
        changed = true;
      }

      return changed;
    }

  public:
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      for (auto &&F: M)
        changed |= runOnGlobal(F);
      for (auto &&G: M.globals())
        changed |= runOnGlobal(G);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

namespace {
  class WeakTypeinfoVTablePass : public PassInfoMixin<WeakTypeinfoVTablePass> {
    bool runOnGlobalVariable(GlobalVariable& GV) {
      // Only need to consider symbols with external linkage because only
      // these could be reported as duplicate.
      if (GV.getLinkage() != llvm::GlobalValue::ExternalLinkage)
        return false;

      if (GV.getName().starts_with("_ZT")) {
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
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      for (auto &&GV : M.globals())
        changed |= runOnGlobalVariable(GV);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

namespace {

  // Add a suffix to the CUDA module ctor/dtor, CUDA specific functions and
  // variables to generate a unique name. This is necessary for lazy
  // compilation. Without suffix, cling cannot distinguish ctor/dtor, register
  // function and and ptx code string of subsequent modules.
  class UniqueCUDAStructorName : public PassInfoMixin<UniqueCUDAStructorName> {
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

      if (GV.getName() == "__cuda_fatbin_wrapper" ||
          GV.getName() == "__cuda_gpubin_handle") {
        GV.setName(add_module_suffix(GV.getName(), ModuleName));
        return true;
      }

      return false;
    }

    // make CUDA specific functions unique
    bool runOnFunction(Function& F, const StringRef ModuleName) {
      if (F.hasName() && (F.getName() == "__cuda_module_ctor" ||
                          F.getName() == "__cuda_module_dtor" ||
                          F.getName() == "__cuda_register_globals")) {
        F.setName(add_module_suffix(F.getName(), ModuleName));
        return true;
      }

      return false;
    }

  public:
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      const StringRef ModuleName = M.getName();
      for (auto&& F : M)
        changed |= runOnFunction(F, ModuleName);
      for (auto&& G : M.globals())
        changed |= runOnGlobal(G, ModuleName);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
} // namespace

namespace {

  // Replace definitions of weak symbols for which symbols already exist by
  // declarations. This reduces the amount of emitted symbols.
  class ReuseExistingWeakSymbols
      : public PassInfoMixin<ReuseExistingWeakSymbols> {
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
    ReuseExistingWeakSymbols(IncrementalJIT& JIT) : m_JIT(JIT) {}

    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      // FIXME: use SymbolLookupSet, rather than looking up symbol by symbol.
      for (auto &&F: M)
        changed |= runOnFunc(F);
      for (auto &&G: M.globals())
        changed |= runOnVar(G);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

// From clang/lib/CodeGen/BackendUtil.cpp
static OptimizationLevel mapToLevel(const CodeGenOptions& Opts) {
  switch (Opts.OptimizationLevel) {
    default: llvm_unreachable("Invalid optimization level!");

    case 0: return OptimizationLevel::O0;

    case 1: return OptimizationLevel::O1;

    case 2:
      switch (Opts.OptimizeSize) {
        default: llvm_unreachable("Invalid optimization level for size!");

        case 0: return OptimizationLevel::O2;

        case 1: return OptimizationLevel::Os;

        case 2: return OptimizationLevel::Oz;
      }

    case 3: return OptimizationLevel::O3;
  }
}

BackendPasses::BackendPasses(const clang::CodeGenOptions &CGOpts,
                             IncrementalJIT &JIT, llvm::TargetMachine& TM):
   m_TM(TM),
   m_JIT(JIT),
   m_CGOpts(CGOpts)
{}


BackendPasses::~BackendPasses() {
  //delete m_PMBuilder->Inliner;
}

void BackendPasses::CreatePasses(int OptLevel, llvm::ModulePassManager& MPM,
                                 llvm::LoopAnalysisManager& LAM,
                                 llvm::FunctionAnalysisManager& FAM,
                                 llvm::CGSCCAnalysisManager& CGAM,
                                 llvm::ModuleAnalysisManager& MAM,
                                 PassInstrumentationCallbacks& PIC,
                                 StandardInstrumentations& SI) {

  // TODO: Remove this pass once we upgrade past LLVM 19 that includes the fix.
  MPM.addPass(WorkAroundConstructorPriorityBugPass());
  MPM.addPass(UniqueInitFunctionNamePass());
  MPM.addPass(KeepLocalGVPass());
  MPM.addPass(WeakTypeinfoVTablePass());
  MPM.addPass(ReuseExistingWeakSymbols(m_JIT));
  MPM.addPass(PreventLocalOptPass());

  // Run verifier after local passes to make sure that IR remains untouched.
  if (m_CGOpts.VerifyModule)
    MPM.addPass(VerifierPass());

  // Handle disabling of LLVM optimization, where we want to preserve the
  // internal module before any optimization.
  if (m_CGOpts.DisableLLVMPasses) {
    // Always keep at least ForceInline - NoInlining is deadly for libc++.
    // Inlining = CGOpts.NoInlining;
    MPM.addPass(AlwaysInlinerPass());
  } else if (OptLevel <= 1) {
    // At O0 and O1 we only run the always inliner which is more efficient. At
    // higher optimization levels we run the normal inliner.
    MPM.addPass(AlwaysInlinerPass());

    // Register a callback for disabling all other inliner passes.
    PIC.registerShouldRunOptionalPassCallback([](StringRef P, Any) {
      if (P == "ModuleInlinerWrapperPass" ||
          P == "InlineAdvisorAnalysisPrinterPass" ||
          P == "PartialInlinerPass" || P == "buildInlinerPipeline" ||
          P == "ModuleInlinerPass" || P == "InlinerPass" ||
          P == "InlineAdvisorAnalysis" || P == "PartiallyInlineLibCallsPass" ||
          P == "RelLookupTableConverterPass" ||
          P == "InlineCostAnnotationPrinterPass" ||
          P == "InlineSizeEstimatorAnalysisPrinterPass" ||
          P == "InlineSizeEstimatorAnalysis")
        return false;

      return true;
    });
  } else {
    // Register a callback for disabling RelLookupTableConverterPass.
    PIC.registerShouldRunOptionalPassCallback(
        [](StringRef P, Any) { return P != "RelLookupTableConverterPass"; });
  }

  SI.registerCallbacks(PIC, &MAM);

  PipelineTuningOptions PTO;
  std::optional<PGOOptions> PGOOpt;
  PassBuilder PB(&m_TM, PTO, PGOOpt, &PIC);

  // Attempt to load pass plugins and register their callbacks with PB.
  for (auto& PluginFN : m_CGOpts.PassPlugins) {
    auto PassPlugin = PassPlugin::Load(PluginFN);
    if (PassPlugin) {
      PassPlugin->registerPassBuilderCallbacks(PB);
    }
  }

  if (!m_CGOpts.DisableLLVMPasses) {
    // Use the default pass pipeline. We also have to map our optimization
    // levels into one of the distinct levels used to configure the pipeline.
    OptimizationLevel Level = mapToLevel(m_CGOpts);
    if (m_CGOpts.OptimizationLevel == 0) {
      // TODO: Remove this after https://reviews.llvm.org/D146200
      MPM.addPass(PB.buildO0DefaultPipeline(Level));
    } else {
      MPM.addPass(PB.buildPerModuleDefaultPipeline(Level));
    }
  }

  // The function __cuda_module_ctor and __cuda_module_dtor will just generated,
  // if a CUDA fatbinary file exist. Without file path there is no need for the
  // function pass.
  if(!m_CGOpts.CudaGpuBinaryFileName.empty())
    MPM.addPass(UniqueCUDAStructorName());

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
}

void BackendPasses::runOnModule(Module& M, int OptLevel) {

  if (OptLevel < 0)
    OptLevel = 0;
  if (OptLevel > 3)
    OptLevel = 3;

  ModulePassManager MPM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI(M.getContext(), m_CGOpts.DebugPassManager);

  CreatePasses(OptLevel, MPM, LAM, FAM, CGAM, MAM, PIC, SI);

  static constexpr std::array<llvm::CodeGenOptLevel, 4> CGOptLevel{
      {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Less,
       llvm::CodeGenOptLevel::Default, llvm::CodeGenOptLevel::Aggressive}};
  // TM's OptLevel is used to build orc::SimpleCompiler passes for every Module.
  m_TM.setOptLevel(CGOptLevel[OptLevel]);

  // Now that we have all of the passes ready, run them.
  MPM.run(M, MAM);
}
