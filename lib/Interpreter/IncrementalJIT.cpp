//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Stefan Gränitz <stefan.graenitz@gmail.com>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IncrementalJIT.h"

// FIXME: Merge IncrementalExecutor and IncrementalJIT.
#include "IncrementalExecutor.h"

#include "cling/Utils/Output.h"
#include "cling/Utils/Utils.h"

#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Frontend/CompilerInstance.h>

#include <llvm/ExecutionEngine/JITLink/EHFrameSupport.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include <optional>

#ifdef __linux__
#include <sys/stat.h>
#endif

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

namespace {

  class ClingMMapper final : public SectionMemoryManager::MemoryMapper {
  public:
    sys::MemoryBlock
    allocateMappedMemory(SectionMemoryManager::AllocationPurpose Purpose,
                         size_t NumBytes,
                         const sys::MemoryBlock* const NearBlock,
                         unsigned Flags, std::error_code& EC) override {
      return sys::Memory::allocateMappedMemory(NumBytes, NearBlock, Flags, EC);
    }

    std::error_code protectMappedMemory(const sys::MemoryBlock& Block,
                                        unsigned Flags) override {
      return sys::Memory::protectMappedMemory(Block, Flags);
    }

    std::error_code releaseMappedMemory(sys::MemoryBlock& M) override {
      // Disabled until CallFunc is informed about unloading, and can
      // re-generate the wrapper (if the decl is still available). See
      // https://github.com/root-project/root/issues/10898
#if 0
      return sys::Memory::releaseMappedMemory(M);
#else
      return {};
#endif
    }
  };

  // A memory manager for Cling that reserves memory for code and data sections
  // to keep them contiguous for the emission of one module. This is required
  // for working exception handling support since one .eh_frame section will
  // refer to many separate .text sections. However, stack unwinding in libgcc
  // assumes that two unwinding objects (for example coming from two modules)
  // are non-overlapping, which is hard to guarantee with separate allocations
  // for the individual code sections.
  class ClingMemoryManager : public SectionMemoryManager {
    using Super = SectionMemoryManager;

    struct AllocInfo {
      uint8_t* m_End = nullptr;
      uint8_t* m_Current = nullptr;

      void setAllocation(uint8_t* Addr, uintptr_t Size) {
        m_Current = Addr;
        m_End = Addr + Size;
      }

      uint8_t* getNextAddr(uintptr_t Size, unsigned Alignment) {
        if (!Alignment)
          Alignment = 16;

        assert(!(Alignment & (Alignment - 1)) &&
               "Alignment must be a power of two.");

        uintptr_t RequiredSize =
            Alignment * ((Size + Alignment - 1) / Alignment + 1);
        if ((m_Current + RequiredSize) > m_End) {
          // This must be the last block.
          if ((m_Current + Size) <= m_End) {
            RequiredSize = Size;
          } else {
            cling::errs()
                << "Error in block allocation by ClingMemoryManager.\n"
                << "Not enough memory was reserved for the current module.\n"
                << Size << " (with alignment: " << RequiredSize
                << " ) is needed but we only have " << (m_End - m_Current)
                << ".\n";
            return nullptr;
          }
        }

        uintptr_t Addr = (uintptr_t)m_Current;

        // Align the address.
        Addr = (Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1);

        m_Current = (uint8_t*)(Addr + Size);

        return (uint8_t*)Addr;
      }

      operator bool() { return m_Current != nullptr; }
    };

    AllocInfo m_Code;
    AllocInfo m_ROData;
    AllocInfo m_RWData;

  public:
    ClingMemoryManager(ClingMMapper& MMapper) : Super(&MMapper) {}

    uint8_t* allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID,
                                 StringRef SectionName) override {
      uint8_t* Addr = nullptr;
      if (m_Code) {
        Addr = m_Code.getNextAddr(Size, Alignment);
      }
      if (!Addr) {
        Addr =
            Super::allocateCodeSection(Size, Alignment, SectionID, SectionName);
      }

      return Addr;
    }

    uint8_t* allocateDataSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID, StringRef SectionName,
                                 bool IsReadOnly) override {

      uint8_t* Addr = nullptr;
      if (IsReadOnly) {
        if (m_ROData) {
          Addr = m_ROData.getNextAddr(Size, Alignment);
        }
      } else if (m_RWData) {
        Addr = m_RWData.getNextAddr(Size, Alignment);
      }
      if (!Addr) {
        Addr = Super::allocateDataSection(Size, Alignment, SectionID,
                                          SectionName, IsReadOnly);
      }
      return Addr;
    }

    void reserveAllocationSpace(uintptr_t CodeSize, Align CodeAlign,
                                uintptr_t RODataSize, Align RODataAlign,
                                uintptr_t RWDataSize,
                                Align RWDataAlign) override {
      m_Code.setAllocation(
          Super::allocateCodeSection(CodeSize, CodeAlign.value(),
                                     /*SectionID=*/0,
                                     /*SectionName=*/"codeReserve"),
          CodeSize);
      m_ROData.setAllocation(
          Super::allocateDataSection(RODataSize, RODataAlign.value(),
                                     /*SectionID=*/0,
                                     /*SectionName=*/"rodataReserve",
                                     /*IsReadOnly=*/true),
          RODataSize);
      m_RWData.setAllocation(
          Super::allocateDataSection(RWDataSize, RWDataAlign.value(),
                                     /*SectionID=*/0,
                                     /*SectionName=*/"rwataReserve",
                                     /*IsReadOnly=*/false),
          RWDataSize);
    }

    bool needsToReserveAllocationSpace() override { return true; }
  };

  /// A JITLinkMemoryManager for Cling that never frees its allocations.
  class ClingJITLinkMemoryManager : public InProcessMemoryManager {
  public:
    using InProcessMemoryManager::InProcessMemoryManager;

    void deallocate(std::vector<FinalizedAlloc> Allocs,
                    OnDeallocatedFunction OnDeallocated) override {
      // Disabled until CallFunc is informed about unloading, and can
      // re-generate the wrapper (if the decl is still available). See
      // https://github.com/root-project/root/issues/10898

      // We still have to release the allocations which resets their addresses
      // to FinalizedAlloc::InvalidAddr, or the assertion in ~FinalizedAlloc
      // will be unhappy...
      for (auto &Alloc : Allocs) {
        Alloc.release();
      }
      // Pretend we successfully deallocated everything...
      OnDeallocated(Error::success());
    }
  };

  /// A DynamicLibrarySearchGenerator that uses ResourceTracker to remember
  /// which symbols were resolved through dlsym during a transaction's reign.
  /// Enables JITDyLib forgetting symbols upon unloading of a shared library.
  /// While JITDylib::define() *is* invoked for these symbols, there is no RT
  /// provided, and thus resource tracking doesn't work, no symbol removal
  /// happens upon unloading the corresponding shared library.
  ///
  /// This might remove more symbols than strictly needed:
  /// 1. libA is loaded
  /// 2. libB is loaded
  /// 3. symbol is resolved from libA
  /// 4. libB is unloaded, removing the symbol, too
  /// That's fine, it will trigger a subsequent dlsym to re-create the symbol.
class RTDynamicLibrarySearchGenerator : public DefinitionGenerator {
public:
  using SymbolPredicate = std::function<bool(const SymbolStringPtr &)>;
  using RTGetterFunc = std::function<ResourceTrackerSP()>;

  /// Create a RTDynamicLibrarySearchGenerator that searches for symbols in the
  /// given sys::DynamicLibrary.
  ///
  /// If the Allow predicate is given then only symbols matching the predicate
  /// will be searched for. If the predicate is not given then all symbols will
  /// be searched for.
  RTDynamicLibrarySearchGenerator(sys::DynamicLibrary Dylib, char GlobalPrefix,
                                  RTGetterFunc RT,
                                  SymbolPredicate Allow = SymbolPredicate());

  /// Permanently loads the library at the given path and, on success, returns
  /// a DynamicLibrarySearchGenerator that will search it for symbol definitions
  /// in the library. On failure returns the reason the library failed to load.
  static Expected<std::unique_ptr<RTDynamicLibrarySearchGenerator>>
  Load(const char *FileName, char GlobalPrefix, RTGetterFunc RT,
       SymbolPredicate Allow = SymbolPredicate());

  /// Creates a RTDynamicLibrarySearchGenerator that searches for symbols in
  /// the current process.
  static Expected<std::unique_ptr<RTDynamicLibrarySearchGenerator>>
  GetForCurrentProcess(char GlobalPrefix, RTGetterFunc RT,
                       SymbolPredicate Allow = SymbolPredicate()) {
    return Load(nullptr, GlobalPrefix, std::move(RT), std::move(Allow));
  }

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Symbols) override;

private:
  sys::DynamicLibrary Dylib;
  RTGetterFunc CurrentRT;
  SymbolPredicate Allow;
  char GlobalPrefix;
};

RTDynamicLibrarySearchGenerator::RTDynamicLibrarySearchGenerator(
    sys::DynamicLibrary Dylib, char GlobalPrefix, RTGetterFunc RT,
    SymbolPredicate Allow)
    : Dylib(std::move(Dylib)), CurrentRT(std::move(RT)),
      Allow(std::move(Allow)), GlobalPrefix(GlobalPrefix) {}

Expected<std::unique_ptr<RTDynamicLibrarySearchGenerator>>
RTDynamicLibrarySearchGenerator::Load(const char *FileName, char GlobalPrefix,
                                      RTGetterFunc RT, SymbolPredicate Allow) {
  std::string ErrMsg;
  auto Lib = sys::DynamicLibrary::getPermanentLibrary(FileName, &ErrMsg);
  if (!Lib.isValid())
    return make_error<StringError>(std::move(ErrMsg), inconvertibleErrorCode());
  return std::make_unique<RTDynamicLibrarySearchGenerator>(
      std::move(Lib), GlobalPrefix, RT, std::move(Allow));
}

Error RTDynamicLibrarySearchGenerator::tryToGenerate(
    LookupState &LS, LookupKind K, JITDylib &JD,
    JITDylibLookupFlags JDLookupFlags, const SymbolLookupSet &Symbols) {
  orc::SymbolMap NewSymbols;

  for (auto &KV : Symbols) {
    auto &Name = KV.first;

    if ((*Name).empty())
      continue;

    if (Allow && !Allow(Name))
      continue;

    bool StripGlobalPrefix = (GlobalPrefix != '\0' && (*Name).front() == GlobalPrefix);

    std::string Tmp((*Name).data() + StripGlobalPrefix,
                    (*Name).size() - StripGlobalPrefix);
    if (void* P = Dylib.getAddressOfSymbol(Tmp.c_str())) {
      NewSymbols[Name] = {orc::ExecutorAddr::fromPtr(P),
                          JITSymbolFlags::Exported};
    }
  }

  if (NewSymbols.empty())
    return Error::success();

  return JD.define(absoluteSymbols(std::move(NewSymbols)), CurrentRT());
}

/// A definition generator that calls a user-provided function that is
/// responsible for providing symbol addresses.
/// This is used by `IncrementalJIT::getGenerator()` to yield a generator that
/// resolves symbols defined in the IncrementalJIT object on which the function
/// is called, which in turn may be used to provide lookup across different
/// IncrementalJIT instances.
class DelegateGenerator : public DefinitionGenerator {
  using LookupFunc = std::function<Expected<llvm::orc::ExecutorAddr>(StringRef)>;
  LookupFunc lookup;

public:
  DelegateGenerator(LookupFunc lookup) : lookup(lookup) {}

  Error tryToGenerate(LookupState& LS, LookupKind K, JITDylib& JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet& LookupSet) override {
    SymbolMap Symbols;
    for (auto& KV : LookupSet) {
      auto Addr = lookup(*KV.first);
      if (auto Err = Addr.takeError())
        return Err;
      Symbols[KV.first] = {Addr.get(), JITSymbolFlags::Exported};
    }
    if (Symbols.empty())
      return Error::success();
    return JD.define(absoluteSymbols(std::move(Symbols)));
  }
};

static bool UseJITLink(const Triple& TT) {
  bool jitLink = false;
  // Default to JITLink on macOS and RISC-V, as done in (recent) LLVM by
  // LLJITBuilderState::prepareForConstruction.
  if (TT.getArch() == Triple::riscv64 || TT.getArch() == Triple::loongarch64 ||
      (TT.isOSBinFormatMachO() &&
       (TT.getArch() == Triple::aarch64 || TT.getArch() == Triple::x86_64)) ||
      (TT.isOSBinFormatELF() &&
       (TT.getArch() == Triple::aarch64 || TT.getArch() == Triple::ppc64le))) {
    jitLink = true;
  }
  // Finally, honor the user's choice by setting an environment variable.
  if (const char* clingJitLink = std::getenv("CLING_JITLINK")) {
    jitLink = cling::utils::ConvertEnvValueToBool(clingJitLink);
  }
  return jitLink;
}

static std::unique_ptr<TargetMachine>
CreateTargetMachine(const clang::CompilerInstance& CI, bool JITLink) {
  CodeGenOptLevel OptLevel = CodeGenOptLevel::Default;
  switch (CI.getCodeGenOpts().OptimizationLevel) {
    case 0: OptLevel = CodeGenOptLevel::None; break;
    case 1: OptLevel = CodeGenOptLevel::Less; break;
    case 2: OptLevel = CodeGenOptLevel::Default; break;
    case 3: OptLevel = CodeGenOptLevel::Aggressive; break;
    default: OptLevel = CodeGenOptLevel::Default;
  }

  const Triple &TT = CI.getTarget().getTriple();

  using namespace llvm::orc;
  auto JTMB = JITTargetMachineBuilder(TT);
  JTMB.addFeatures(CI.getTargetOpts().Features);
  JTMB.getOptions().MCOptions.ABIName = CI.getTarget().getABI().str();

  JTMB.setCodeGenOptLevel(OptLevel);
#ifdef _WIN32
  JTMB.getOptions().EmulatedTLS = false;
#endif // _WIN32

#if defined(__powerpc64__) || defined(__PPC64__)
  // We have to use large code model for PowerPC64 because TOC and text sections
  // can be more than 2GB apart.
  JTMB.setCodeModel(CodeModel::Large);
#endif

  if (JITLink) {
    // Set up the TargetMachine as otherwise done by
    // LLJITBuilderState::prepareForConstruction.
    JTMB.setRelocationModel(Reloc::PIC_);
    // However, do not change the code model: While the small code model would
    // be enough if we created a new JITDylib per Module, Cling adds multiple
    // modules to the same library. This causes problems because weak symbols
    // are merged and may end up more than 2 GB apart. This is especially
    // visible for DW.ref.__gxx_personality_v0 related to exception handling.
    // TODO: Investigate if we can use one JITDylib per Module as recommended
    // by upstream.
  }

  return cantFail(JTMB.createTargetMachine());
}

#ifdef __APPLE__
// Forward-declare compiler-rt complex division helpers
extern "C" {
void __divsc3();
void __divdc3();
}

static SymbolMap GetListOfMacOSCompilerRTSymbols(const LLJIT& Jit) {
  // Inject symbols that may not be resolved while JIT'ing on macOS
  static const std::pair<const char*, const void*> NamePtrList[] = {
      {"__divsc3", (void*)&__divsc3},
      {"__divdc3", (void*)&__divdc3},
  };
  SymbolMap CompilerRTSymbols;
  for (const auto& NamePtr : NamePtrList) {
    CompilerRTSymbols[Jit.mangleAndIntern(NamePtr.first)] = {
        orc::ExecutorAddr::fromPtr(NamePtr.second), JITSymbolFlags::Exported};
  }
  return CompilerRTSymbols;
}
#endif

#if defined(__linux__) && defined(__GLIBC__)
static SymbolMap GetListOfLibcNonsharedSymbols(const LLJIT& Jit) {
  // Inject a number of symbols that may be in libc_nonshared.a where they are
  // not found automatically. Before DefinitionGenerators in ORCv2, this used
  // to be done by RTDyldMemoryManager::getSymbolAddressInProcess See also the
  // upstream issue https://github.com/llvm/llvm-project/issues/61289.

  static const std::pair<const char*, const void*> NamePtrList[] = {
      {"stat", (void*)&stat},       {"fstat", (void*)&fstat},
      {"lstat", (void*)&lstat},     {"stat64", (void*)&stat64},
      {"fstat64", (void*)&fstat64}, {"lstat64", (void*)&lstat64},
      {"fstatat", (void*)&fstatat}, {"fstatat64", (void*)&fstatat64},
      {"mknod", (void*)&mknod},     {"mknodat", (void*)&mknodat},
  };

  SymbolMap LibcNonsharedSymbols;
  for (const auto& NamePtr : NamePtrList) {
    LibcNonsharedSymbols[Jit.mangleAndIntern(NamePtr.first)] = {
        orc::ExecutorAddr::fromPtr(NamePtr.second), JITSymbolFlags::Exported};
  }
  return LibcNonsharedSymbols;
}
#endif
} // unnamed namespace

namespace cling {

///\brief Creates JIT event listener to allow profiling of JITted code with perf
llvm::JITEventListener* createPerfJITEventListener();

IncrementalJIT::~IncrementalJIT() {
  // FIXME: This should ideally happen in the right order without explicitly
  // doing this. We started seeing failing tests (eg, tutorial-hist-cumulative,
  // JITLink turned on) with assertion failure in ~FinalizedAlloc after commit
  // [cling] Move generators to ProcessSymbols JITDylib
  // This likely changed the destruction order that caused the assertion to
  // trigger.
  if (auto Err = Jit->getMainJITDylib().clear()) {
    llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                "Error clearing MainJITDylib: ");
  }

  if (auto Err = Jit->getProcessSymbolsJITDylib()->clear()) {
    llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                "Error clearing ProcessSymbolsJITDylib: ");
  }
}

IncrementalJIT::IncrementalJIT(
    IncrementalExecutor& Executor, const clang::CompilerInstance &CI,
    std::unique_ptr<llvm::orc::ExecutorProcessControl> EPC, Error& Err,
    void *ExtraLibHandle, bool Verbose)
    : SkipHostProcessLookup(false),
      m_JITLink(UseJITLink(CI.getTarget().getTriple())),
      m_TM(CreateTargetMachine(CI, m_JITLink)),
      SingleThreadedContext(std::make_unique<LLVMContext>()) {
  ErrorAsOutParameter _(&Err);

  LLJITBuilder Builder;
  Builder.setDataLayout(m_TM->createDataLayout());
  Builder.setExecutorProcessControl(std::move(EPC));

  // Create ObjectLinkingLayer with our own MemoryManager.
  Builder.setObjectLinkingLayerCreator([&](ExecutionSession& ES,
                                           const Triple& TT)
                                           -> std::unique_ptr<ObjectLayer> {
    if (m_JITLink) {
      // For JITLink, we only need a custom memory manager to avoid freeing the
      // memory segments; the default InProcessMemoryManager (which is mostly
      // copied above) already does slab allocation to keep all segments
      // together which is needed for exception handling support.
      unsigned PageSize = cantFail(sys::Process::getPageSize());
      auto ObjLinkingLayer = std::make_unique<ObjectLinkingLayer>(
          ES, std::make_unique<ClingJITLinkMemoryManager>(PageSize));
      ObjLinkingLayer->addPlugin(std::make_unique<EHFrameRegistrationPlugin>(
          ES, std::make_unique<InProcessEHFrameRegistrar>()));
      return ObjLinkingLayer;
    }

    auto MMapper = std::make_unique<ClingMMapper>();
    auto GetMemMgr = [MMapper = std::move(MMapper)]() {
      return std::make_unique<ClingMemoryManager>(*MMapper);
    };
    auto Layer =
        std::make_unique<RTDyldObjectLinkingLayer>(ES, std::move(GetMemMgr));

    // Register JIT event listeners if enabled
    if (cling::utils::ConvertEnvValueToBool(std::getenv("CLING_DEBUG")))
      Layer->registerJITEventListener(
          *JITEventListener::createGDBRegistrationListener());

#ifdef __linux__
    if (cling::utils::ConvertEnvValueToBool(std::getenv("CLING_PROFILE")))
      Layer->registerJITEventListener(*cling::createPerfJITEventListener());
#endif

    // The following is based on LLJIT::createObjectLinkingLayer.
    if (TT.isOSBinFormatCOFF()) {
      Layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      Layer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    if (TT.isOSBinFormatELF() && (TT.getArch() == Triple::ArchType::aarch64 ||
                                  TT.getArch() == Triple::ArchType::ppc64 ||
                                  TT.getArch() == Triple::ArchType::ppc64le))
      Layer->setAutoClaimResponsibilityForObjectSymbols(true);

    return Layer;
  });

  Builder.setCompileFunctionCreator([&](llvm::orc::JITTargetMachineBuilder)
  -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    return std::make_unique<SimpleCompiler>(*m_TM);
  });

  char LinkerPrefix = this->m_TM->createDataLayout().getGlobalPrefix();

  Builder.setProcessSymbolsJITDylibSetup([&](LLJIT& J) -> Expected<JITDylibSP> {
    auto& JD = J.getExecutionSession().createBareJITDylib("<Process Symbols>");
    // Process symbol resolution
    auto HostProcessLookup =
        RTDynamicLibrarySearchGenerator::GetForCurrentProcess(
            LinkerPrefix, [this] { return m_CurrentProcessRT; },
            [this](const SymbolStringPtr& Sym) {
              return !m_ForbidDlSymbols.contains(*Sym);
            });
    if (!HostProcessLookup) {
      return HostProcessLookup.takeError();
    }
    JD.addGenerator(std::move(*HostProcessLookup));

    // This must come after process resolution, to  consistently resolve global
    // symbols (e.g. std::cout) to the same address.
    auto LibLookup = std::make_unique<RTDynamicLibrarySearchGenerator>(
        llvm::sys::DynamicLibrary(ExtraLibHandle), LinkerPrefix,
        [this] { return m_CurrentProcessRT; },
        [this](const SymbolStringPtr& Sym) {
          return !m_ForbidDlSymbols.contains(*Sym);
        });
    JD.addGenerator(std::move(LibLookup));
    return &JD;
  });

  if (Expected<std::unique_ptr<LLJIT>> JitInstance = Builder.create()) {
    Jit = std::move(*JitInstance);
  } else {
    Err = JitInstance.takeError();
    return;
  }

  // We use this callback to transfer the ownership of the ThreadSafeModule,
  // which owns the Transaction's llvm::Module, to m_CompiledModules.
  Jit->getIRCompileLayer().setNotifyCompiled([this](auto &MR,
                                                    ThreadSafeModule TSM) {
      // FIXME: Don't store them mapped by raw pointers.
      const Module *Unsafe = TSM.getModuleUnlocked();
      assert(!m_CompiledModules.count(Unsafe) && "Modules are compiled once");
      m_CompiledModules[Unsafe] = std::move(TSM);
    });

#if defined(__linux__) && defined(__GLIBC__)
  // See comment in ListOfLibcNonsharedSymbols.
  cantFail(Jit->getProcessSymbolsJITDylib()->define(
      absoluteSymbols(GetListOfLibcNonsharedSymbols(*Jit))));
#endif

#if defined(__APPLE__)
  cantFail(Jit->getProcessSymbolsJITDylib()->define(
      absoluteSymbols(GetListOfMacOSCompilerRTSymbols(*Jit))));
#endif

  // This replaces llvm::orc::ExecutionSession::logErrorsToStdErr:
  auto&& ErrorReporter = [&Executor, LinkerPrefix, Verbose](Error Err) {
    Err = handleErrors(std::move(Err),
                       [&](std::unique_ptr<SymbolsNotFound> Err) -> Error {
                         // IncrementalExecutor has its own diagnostics (for
                         // now) that tries to guess which library needs to be
                         // loaded.
                         for (auto&& symbol : Err->getSymbols()) {
                           std::string symbolStr = (*symbol).str();
                           if (LinkerPrefix != '\0' &&
                               symbolStr[0] == LinkerPrefix) {
                             symbolStr.erase(0, 1);
                           }
                           Executor.HandleMissingFunction(symbolStr);
                         }

                         // However, the diagnstic here might be superior as
                         // they show *all* unresolved symbols, so show them in
                         // case of "verbose" nonetheless.
                         if (Verbose)
                           return Error(std::move(Err));
                         return Error::success();
                       });

    if (!Err)
      return;

    logAllUnhandledErrors(std::move(Err), errs(), "cling JIT session error: ");
  };
  Jit->getExecutionSession().setErrorReporter(ErrorReporter);
}

std::unique_ptr<llvm::orc::DefinitionGenerator> IncrementalJIT::getGenerator() {
  return std::make_unique<DelegateGenerator>(
      [&](StringRef Name) { return Jit->lookupLinkerMangled(Name); });
}

void IncrementalJIT::addModule(Transaction& T) {
  ResourceTrackerSP MainRT = Jit->getMainJITDylib().createResourceTracker();
  m_MainResourceTrackers[&T] = MainRT;
  ResourceTrackerSP ProcessRT =
      Jit->getProcessSymbolsJITDylib()->createResourceTracker();
  m_ProcessResourceTrackers[&T] = ProcessRT;

  std::unique_ptr<Module> module = T.takeModule();

  // Reset the sections of all functions so that they end up in the same text
  // section. This is important for TCling on macOS to catch exceptions raised
  // by constructors, which requires unwinding information. The addresses in
  // the __eh_frame table are relocated against a single __text section when
  // loading the MachO binary, which breaks if the call sites of constructors
  // end up in a separate init section.
  // (see clang::TargetInfo::getStaticInitSectionSpecifier())
  for (auto &Fn : module->functions()) {
    if (Fn.hasSection()) {
      // dbgs() << "Resetting section '" << Fn.getSection() << "' of function "
      //        << Fn.getName() << "\n";
      Fn.setSection("");
    }
  }

  ThreadSafeModule TSM(std::move(module), SingleThreadedContext);

  const Module *Unsafe = TSM.getModuleUnlocked();
  T.m_CompiledModule = Unsafe;
  m_CurrentProcessRT = ProcessRT;

  if (Error Err = Jit->addIRModule(MainRT, std::move(TSM))) {
    logAllUnhandledErrors(std::move(Err), errs(),
                          "[IncrementalJIT] addModule() failed: ");
    return;
  }
}

llvm::Error IncrementalJIT::removeModule(const Transaction& T) {
  ResourceTrackerSP MainRT = std::move(m_MainResourceTrackers[&T]);
  if (!MainRT)
    return llvm::Error::success();
  ResourceTrackerSP ProcessRT = std::move(m_ProcessResourceTrackers[&T]);

  m_MainResourceTrackers.erase(&T);
  m_ProcessResourceTrackers.erase(&T);
  if (Error Err = MainRT->remove())
    return Err;
  if (Error Err = ProcessRT->remove())
    return Err;
  auto iMod = m_CompiledModules.find(T.m_CompiledModule);
  if (iMod != m_CompiledModules.end())
    m_CompiledModules.erase(iMod);

  return llvm::Error::success();
}

orc::ExecutorAddr
IncrementalJIT::addOrReplaceDefinition(StringRef Name,
                                       orc::ExecutorAddr KnownAddr) {

  // Let's inject it
  bool Inserted;
  SymbolMap::iterator It;
  std::tie(It, Inserted) = m_InjectedSymbols.try_emplace(
      Jit->mangleAndIntern(Name),
      ExecutorSymbolDef(KnownAddr, JITSymbolFlags::Exported));
  assert(Inserted && "Why wasn't this found in the initial Jit lookup?");

  bool Defined = false;
  for (auto* Dylib :
       {&Jit->getMainJITDylib(), Jit->getPlatformJITDylib().get()}) {
    if (Error Err = Dylib->remove({It->first})) {
      Err = handleErrors(std::move(Err),
                         [&](std::unique_ptr<SymbolsNotFound> Err) -> Error {
                           // This is fine, we will try in the next Dylib.
                           return Error::success();
                         });

      if (Err) {
        logAllUnhandledErrors(std::move(Err), errs(),
                              "[IncrementalJIT] remove() failed: ");
        return orc::ExecutorAddr();
      }

      continue;
    }

    if (Error Err = Dylib->define(absoluteSymbols({*It}))) {
      logAllUnhandledErrors(std::move(Err), errs(),
                            "[IncrementalJIT] define() failed: ");
      return orc::ExecutorAddr();
    }
    Defined = true;
  }

  if (!Defined) {
    // Symbol was not found, just define it in the main library.
    if (Error Err = Jit->getMainJITDylib().define(absoluteSymbols({*It}))) {
      logAllUnhandledErrors(std::move(Err), errs(),
                            "[IncrementalJIT] define() failed: ");
      return orc::ExecutorAddr();
    }
  }

  return KnownAddr;
}

void* IncrementalJIT::getSymbolAddress(StringRef Name, bool IncludeHostSymbols){
  std::unique_lock<SharedAtomicFlag> G(SkipHostProcessLookup, std::defer_lock);
  if (!IncludeHostSymbols)
    G.lock();

  std::pair<llvm::StringMapIterator<std::nullopt_t>, bool> insertInfo;
  if (!IncludeHostSymbols)
    insertInfo = m_ForbidDlSymbols.insert(Name);

  Expected<llvm::orc::ExecutorAddr> Symbol =
      Jit->lookup(Jit->getMainJITDylib(), Name);
  if (!Symbol) {
    consumeError(Symbol.takeError());
    // FIXME: We should take advantage of the fact that all process symbols
    // are now in a separate JITDylib; see also the comments and ideas in
    // IncrementalExecutor::getAddressOfGlobal().
    Symbol = Jit->lookup(*Jit->getProcessSymbolsJITDylib(), Name);
  }

  // If m_ForbidDlSymbols already contained Name before we tried to insert it
  // then some calling frame has added it and will remove it later because its
  // insertInfo.second is true.
  if (!IncludeHostSymbols && insertInfo.second)
    m_ForbidDlSymbols.erase(insertInfo.first);

  if (!Symbol) {
    // This interface is allowed to return nullptr on a missing symbol without
    // diagnostics.
    consumeError(Symbol.takeError());
    return nullptr;
  }

  return (Symbol.get()).toPtr<void*>();
}

bool IncrementalJIT::doesSymbolAlreadyExist(StringRef UnmangledName) {
  auto Name = Jit->mangle(UnmangledName);
  for (auto &&M: m_CompiledModules) {
    if (M.first->getNamedValue(Name))
      return true;
  }
  return false;
}

} // namespace cling
