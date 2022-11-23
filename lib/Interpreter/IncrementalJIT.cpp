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

#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/raw_ostream.h>

#include <sstream>

using namespace llvm;
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

  ClingMMapper MMapperInstance;

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
    ClingMemoryManager() : Super(&MMapperInstance) {}

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

    void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                                uintptr_t RODataSize, uint32_t RODataAlign,
                                uintptr_t RWDataSize,
                                uint32_t RWDataAlign) override {
      m_Code.setAllocation(
          Super::allocateCodeSection(CodeSize, CodeAlign,
                                     /*SectionID=*/0,
                                     /*SectionName=*/"codeReserve"),
          CodeSize);
      m_ROData.setAllocation(
          Super::allocateDataSection(RODataSize, RODataAlign,
                                     /*SectionID=*/0,
                                     /*SectionName=*/"rodataReserve",
                                     /*IsReadOnly=*/true),
          RODataSize);
      m_RWData.setAllocation(
          Super::allocateDataSection(RWDataSize, RWDataAlign,
                                     /*SectionID=*/0,
                                     /*SectionName=*/"rwataReserve",
                                     /*IsReadOnly=*/false),
          RWDataSize);
    }

    bool needsToReserveAllocationSpace() override { return true; }
  };

} // unnamed namespace

namespace cling {

IncrementalJIT::IncrementalJIT(
    IncrementalExecutor& Executor, std::unique_ptr<TargetMachine> TM,
    std::unique_ptr<llvm::orc::ExecutorProcessControl> EPC, Error& Err,
    void *ExtraLibHandle, bool Verbose)
    : SkipHostProcessLookup(false),
      TM(std::move(TM)),
      SingleThreadedContext(std::make_unique<LLVMContext>()) {
  ErrorAsOutParameter _(&Err);

  // FIXME: We should probably take codegen settings from the CompilerInvocation
  // and not from the target machine
  JITTargetMachineBuilder JTMB(this->TM->getTargetTriple());
  JTMB.setCodeModel(this->TM->getCodeModel());
  JTMB.setCodeGenOptLevel(this->TM->getOptLevel());
  JTMB.setFeatures(this->TM->getTargetFeatureString());
  JTMB.setRelocationModel(this->TM->getRelocationModel());

  LLJITBuilder Builder;
  Builder.setJITTargetMachineBuilder(std::move(JTMB));
  Builder.setExecutorProcessControl(std::move(EPC));

  // Create ObjectLinkingLayer with our own MemoryManager.
  Builder.setObjectLinkingLayerCreator([&](ExecutionSession& ES,
                                           const Triple& TT) {
    auto GetMemMgr = []() { return std::make_unique<ClingMemoryManager>(); };
    auto Layer =
        std::make_unique<RTDyldObjectLinkingLayer>(ES, std::move(GetMemMgr));

    // The following is based on LLJIT::createObjectLinkingLayer.
    if (TT.isOSBinFormatCOFF()) {
      Layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      Layer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    if (TT.isOSBinFormatELF() && (TT.getArch() == Triple::ArchType::ppc64 ||
                                  TT.getArch() == Triple::ArchType::ppc64le))
      Layer->setAutoClaimResponsibilityForObjectSymbols(true);

    return Layer;
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

  char LinkerPrefix = this->TM->createDataLayout().getGlobalPrefix();

  // Process symbol resolution
  auto HostProcessLookup = DynamicLibrarySearchGenerator::GetForCurrentProcess(
                                                                  LinkerPrefix,
                                              [&](const SymbolStringPtr &Sym) {
                                  return !m_ForbidDlSymbols.contains(*Sym); });
  if (!HostProcessLookup) {
    Err = HostProcessLookup.takeError();
    return;
  }
  Jit->getMainJITDylib().addGenerator(std::move(*HostProcessLookup));

  // This must come after process resolution, to  consistently resolve global
  // symbols (e.g. std::cout) to the same address.
  auto LibLookup = std::make_unique<DynamicLibrarySearchGenerator>(
                       llvm::sys::DynamicLibrary(ExtraLibHandle), LinkerPrefix,
                                              [&](const SymbolStringPtr &Sym) {
                                  return !m_ForbidDlSymbols.contains(*Sym); });
  Jit->getMainJITDylib().addGenerator(std::move(LibLookup));

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

void IncrementalJIT::addModule(Transaction& T) {
  ResourceTrackerSP RT = Jit->getMainJITDylib().createResourceTracker();
  m_ResourceTrackers[&T] = RT;

  std::ostringstream sstr;
  sstr << T.getModule()->getModuleIdentifier() << '-' << std::hex
       << std::showbase << (size_t)&T;
  ThreadSafeModule TSM(T.takeModule(), SingleThreadedContext);

  const Module *Unsafe = TSM.getModuleUnlocked();
  T.m_CompiledModule = Unsafe;

  if (Error Err = Jit->addIRModule(RT, std::move(TSM))) {
    logAllUnhandledErrors(std::move(Err), errs(),
                          "[IncrementalJIT] addModule() failed: ");
    return;
  }
}

llvm::Error IncrementalJIT::removeModule(const Transaction& T) {
  ResourceTrackerSP RT = std::move(m_ResourceTrackers[&T]);
  if (!RT)
    return llvm::Error::success();

  m_ResourceTrackers.erase(&T);
  if (Error Err = RT->remove())
    return Err;
  return llvm::Error::success();
}

JITTargetAddress
IncrementalJIT::addOrReplaceDefinition(StringRef Name,
                                       JITTargetAddress KnownAddr) {

  void* Symbol = getSymbolAddress(Name, /*IncludeFromHost=*/true);

  // Nothing to define, we are redefining the same function. FIXME: Diagnose.
  if (Symbol && (JITTargetAddress)Symbol == KnownAddr)
    return KnownAddr;

  llvm::SmallString<128> LinkerMangledName;
  char LinkerPrefix = this->TM->createDataLayout().getGlobalPrefix();
  bool HasLinkerPrefix = LinkerPrefix != '\0';
  if (HasLinkerPrefix && Name.front() == LinkerPrefix) {
    LinkerMangledName.assign(1, LinkerPrefix);
    LinkerMangledName.append(Name);
  } else {
    LinkerMangledName.assign(Name);
  }

  // Let's inject it
  bool Inserted;
  SymbolMap::iterator It;
  std::tie(It, Inserted) = m_InjectedSymbols.try_emplace(
      Jit->getExecutionSession().intern(LinkerMangledName),
      JITEvaluatedSymbol(KnownAddr, JITSymbolFlags::Exported));
  assert(Inserted && "Why wasn't this found in the initial Jit lookup?");

  JITDylib& DyLib = Jit->getMainJITDylib();
  // We want to replace a symbol with a custom provided one.
  if (Symbol && KnownAddr)
     // The symbol be in the DyLib or in-process.
     llvm::consumeError(DyLib.remove({It->first}));

  if (Error Err = DyLib.define(absoluteSymbols({*It}))) {
    logAllUnhandledErrors(std::move(Err), errs(),
                          "[IncrementalJIT] define() failed: ");
    return JITTargetAddress{};
  }

  return KnownAddr;
}

void* IncrementalJIT::getSymbolAddress(StringRef Name, bool IncludeHostSymbols) {
  std::unique_lock<SharedAtomicFlag> G(SkipHostProcessLookup, std::defer_lock);
  if (!IncludeHostSymbols)
    G.lock();

  std::pair<llvm::StringMapIterator<llvm::NoneType>, bool> insertInfo;
  if (!IncludeHostSymbols)
    insertInfo = m_ForbidDlSymbols.insert(Name);

  Expected<JITEvaluatedSymbol> Symbol = Jit->lookup(Name);

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

  return jitTargetAddressToPointer<void*>(Symbol->getAddress());
}

} // namespace cling
