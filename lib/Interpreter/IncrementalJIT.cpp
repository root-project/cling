//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IncrementalJIT.h"

#include "IncrementalExecutor.h"
#include "cling/Utils/Platform.h"

#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/DynamicLibrary.h"

#if defined(__APPLE__) || defined (_MSC_VER)
// Apple and Windows add an extra '_'
# define MANGLE_PREFIX "_"
#endif

using namespace llvm;

namespace {

///\brief Memory manager providing the lop-level link to the
/// IncrementalExecutor, handles missing or special / replaced symbols.
class ClingMemoryManager: public SectionMemoryManager {
public:
  ClingMemoryManager(cling::IncrementalExecutor& Exe) {}

  ///\brief Simply wraps the base class's function setting AbortOnFailure
  /// to false and instead using the error handling mechanism to report it.
  void* getPointerToNamedFunction(const std::string &Name,
                                  bool /*AbortOnFailure*/ =true) override {
    return SectionMemoryManager::getPointerToNamedFunction(Name, false);
  }
};

  class NotifyFinalizedT {
  public:
    NotifyFinalizedT(cling::IncrementalJIT &jit) : m_JIT(jit) {}
    void operator()(llvm::orc::RTDyldObjectLinkingLayerBase::ObjHandleT H) {
      m_JIT.RemoveUnfinalizedSection(H);
    }

  private:
    cling::IncrementalJIT &m_JIT;
  };

} // unnamed namespace

namespace cling {

///\brief Memory manager for the OrcJIT layers to resolve symbols from the
/// common IncrementalJIT. I.e. the master of the Orcs.
/// Each ObjectLayer instance has one Azog object.
class Azog: public RTDyldMemoryManager {
  cling::IncrementalJIT& m_jit;

  struct AllocInfo {
    uint8_t *m_Start   = nullptr;
    uint8_t *m_End     = nullptr;
    uint8_t *m_Current = nullptr;

    void allocate(RTDyldMemoryManager *exeMM,
                  uintptr_t Size, uint32_t Align,
                  bool code, bool isReadOnly) {

      uintptr_t RequiredSize = Size;
      if (code)
        m_Start = exeMM->allocateCodeSection(RequiredSize, Align,
                                             0 /* SectionID */,
                                             "codeReserve");
      else if (isReadOnly)
        m_Start = exeMM->allocateDataSection(RequiredSize, Align,
                                             0 /* SectionID */,
                                             "rodataReserve",isReadOnly);
      else
        m_Start = exeMM->allocateDataSection(RequiredSize, Align,
                                             0 /* SectionID */,
                                             "rwataReserve",isReadOnly);
      m_Current = m_Start;
      m_End = m_Start + RequiredSize;
    }

    uint8_t* getNextAddr(uintptr_t Size, unsigned Alignment) {
      if (!Alignment)
        Alignment = 16;

      assert(!(Alignment & (Alignment - 1)) && "Alignment must be a power of two.");

      uintptr_t RequiredSize = Alignment * ((Size + Alignment - 1)/Alignment + 1);
      if ( (m_Current + RequiredSize) > m_End ) {
        // This must be the last block.
        if ((m_Current + Size) <= m_End) {
          RequiredSize = Size;
        } else {
          cling::errs() << "Error in block allocation by Azog. "
                        << "Not enough memory was reserved for the current module. "
                        << Size << " (with alignment: " << RequiredSize
                        << " ) is needed but\n"
                        << "we only have " << (m_End - m_Current) << ".\n";
          return nullptr;
        }
      }

      uintptr_t Addr = (uintptr_t)m_Current;

      // Align the address.
      Addr = (Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1);

      m_Current = (uint8_t*)(Addr + Size);

      return (uint8_t*)Addr;
    }

    operator bool() {
      return m_Current != nullptr;
    }
  };

  AllocInfo m_Code;
  AllocInfo m_ROData;
  AllocInfo m_RWData;

#ifdef CLING_WIN_SEH_EXCEPTIONS
  uintptr_t getBaseAddr() const {
    if (LLVM_LIKELY(m_Code.m_Start && m_ROData.m_Start && m_RWData.m_Start)) {
      return uintptr_t(std::min(std::min(m_Code.m_Start, m_ROData.m_Start),
                                m_RWData.m_Start));
    }
    if (LLVM_LIKELY(m_Code.m_Start)) {
      return uintptr_t(m_ROData.m_Start
                           ? std::min(m_Code.m_Start, m_ROData.m_Start)
                           : std::min(m_Code.m_Start, m_RWData.m_Start));
    }
    return uintptr_t(m_ROData.m_Start && m_RWData.m_Start
                         ? std::min(m_ROData.m_Start, m_RWData.m_Start)
                         : std::max(m_ROData.m_Start, m_RWData.m_Start));
  }

  // FIXME: This is directly mirroring a structure in RTDyldMemoryManager that
  // is private. Get Win64 exceptions into LLVM or add an accessor for it.
  platform::windows::EHFrameInfos m_EHFrames;
#endif

public:
  Azog(cling::IncrementalJIT& Jit): m_jit(Jit) {}

  RTDyldMemoryManager* getExeMM() const { return m_jit.m_ExeMM.get(); }

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override {
    uint8_t *Addr = nullptr;
    if (m_Code) {
      Addr = m_Code.getNextAddr(Size, Alignment);
    }
    if (!Addr) {
      Addr = getExeMM()->allocateCodeSection(Size, Alignment, SectionID, SectionName);
      m_jit.m_SectionsAllocatedSinceLastLoad.insert(Addr);
    }

    return Addr;
  }

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool IsReadOnly) override {

    uint8_t *Addr = nullptr;
    if (IsReadOnly && m_ROData) {
      Addr = m_ROData.getNextAddr(Size,Alignment);
    } else if (m_RWData) {
      Addr = m_RWData.getNextAddr(Size,Alignment);
    }
    if (!Addr) {
      Addr = getExeMM()->allocateDataSection(Size, Alignment, SectionID,
                                                   SectionName, IsReadOnly);
      m_jit.m_SectionsAllocatedSinceLastLoad.insert(Addr);
    }
    return Addr;
  }

  void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                              uintptr_t RODataSize, uint32_t RODataAlign,
                              uintptr_t RWDataSize, uint32_t RWDataAlign) override {
    m_Code.allocate(getExeMM(),CodeSize, CodeAlign, true, false);
    m_ROData.allocate(getExeMM(),RODataSize, RODataAlign, false, true);
    m_RWData.allocate(getExeMM(),RWDataSize, RWDataAlign, false, false);

    m_jit.m_SectionsAllocatedSinceLastLoad.insert(m_Code.m_Start);
    m_jit.m_SectionsAllocatedSinceLastLoad.insert(m_ROData.m_Start);
    m_jit.m_SectionsAllocatedSinceLastLoad.insert(m_RWData.m_Start);
  }

  bool needsToReserveAllocationSpace() override {
    return true; // getExeMM()->needsToReserveAllocationSpace();
  }

  void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                        size_t Size) override {
#ifdef CLING_WIN_SEH_EXCEPTIONS
    const platform::windows::RuntimePRFunction PRFunc = { Addr, Size };
    m_EHFrames.emplace_back(PRFunc);
#else
    return getExeMM()->registerEHFrames(Addr, LoadAddr, Size);
#endif
  }

  void deregisterEHFrames() override {
#ifdef CLING_WIN_SEH_EXCEPTIONS
    platform::DeRegisterEHFrames(getBaseAddr(), m_EHFrames);
    platform::windows::EHFrameInfos().swap(m_EHFrames);
#else
    return getExeMM()->deregisterEHFrames();
#endif
  }

  uint64_t getSymbolAddress(const std::string &Name) override {
    // FIXME: We should decide if we want to handle the error here or make the
    // return type of the function llvm::Expected<uint64_t> relying on the
    // users to decide how to handle the error.
    if (auto Addr = m_jit.getSymbolAddressWithoutMangling(Name,
                                                        true /*also use dlsym*/)
        .getAddress())
      return *Addr;

    llvm_unreachable("Handle the error case");
    return ~0U;
  }

  void *getPointerToNamedFunction(const std::string &Name,
                                  bool AbortOnFailure = true) override {
    return getExeMM()->getPointerToNamedFunction(Name, AbortOnFailure);
  }

  using llvm::RuntimeDyld::MemoryManager::notifyObjectLoaded;

  void notifyObjectLoaded(ExecutionEngine *EE,
                          const object::ObjectFile &O) override {
    return getExeMM()->notifyObjectLoaded(EE, O);
  }

  bool finalizeMemory(std::string *ErrMsg = nullptr) override {
    // Each set of objects loaded will be finalized exactly once, but since
    // symbol lookup during relocation may recursively trigger the
    // loading/relocation of other modules, and since we're forwarding all
    // finalizeMemory calls to a single underlying memory manager, we need to
    // defer forwarding the call on until all necessary objects have been
    // loaded. Otherwise, during the relocation of a leaf object, we will end
    // up finalizing memory, causing a crash further up the stack when we
    // attempt to apply relocations to finalized memory.
    // To avoid finalizing too early, look at how many objects have been
    // loaded but not yet finalized. This is a bit of a hack that relies on
    // the fact that we're lazily emitting object files: The only way you can
    // get more than one set of objects loaded but not yet finalized is if
    // they were loaded during relocation of another set.
    if (m_jit.m_UnfinalizedSections.size() == 1) {
#ifdef CLING_WIN_SEH_EXCEPTIONS
      platform::RegisterEHFrames(getBaseAddr(), m_EHFrames, true);
#endif
      return getExeMM()->finalizeMemory(ErrMsg);
    }
    return false;
  };

}; // class Azog

IncrementalJIT::IncrementalJIT(IncrementalExecutor& exe,
                               std::unique_ptr<TargetMachine> TM):
  m_Parent(exe),
  m_TM(std::move(TM)),
  m_TMDataLayout(m_TM->createDataLayout()),
  m_ExeMM(std::make_shared<ClingMemoryManager>(m_Parent)),
  m_NotifyObjectLoaded(*this),
  m_ObjectLayer(m_SymbolMap, [this] () { return llvm::make_unique<Azog>(*this); },
                m_NotifyObjectLoaded, NotifyFinalizedT(*this)),
  m_CompileLayer(m_ObjectLayer, llvm::orc::SimpleCompiler(*m_TM)),
  m_LazyEmitLayer(m_CompileLayer) {

  // Force the JIT to query for symbols local to itself, i.e. if it resides in a
  // shared library it will resolve symbols from there first. This is done to
  // implement our proto symbol versioning protection. Namely, if some other
  // library provides llvm symbols, we want out JIT to avoid looking at them.
  //
  // FIXME: In general, this approach causes numerous issues when cling is
  // embedded and the framework needs to provide its own set of symbols which
  // exist in llvm. Most notably if the framework links against different
  // versions of linked against llvm libraries. For instance, if we want to provide
  // a custom zlib in the framework the JIT will still resolve to llvm's version
  // of libz causing hard-to-debug bugs. In order to work around such cases we
  // need to swap the llvm system libraries, which can be tricky for two
  // reasons: (a) llvm's cmake doesn't really support it; (b) only works if we
  // build llvm from sources.
  llvm::sys::DynamicLibrary::SearchOrder
    = llvm::sys::DynamicLibrary::SO_LoadedFirst;
  // Enable JIT symbol resolution from the binary.
  llvm::sys::DynamicLibrary::LoadLibraryPermanently(0, 0);

  // Make debug symbols available.
  m_GDBListener = 0; // JITEventListener::createGDBRegistrationListener();

// #if MCJIT
//   llvm::EngineBuilder builder(std::move(m));

//   std::string errMsg;
//   builder.setErrorStr(&errMsg);
//   builder.setOptLevel(llvm::CodeGenOpt::Less);
//   builder.setEngineKind(llvm::EngineKind::JIT);
//   std::unique_ptr<llvm::RTDyldMemoryManager>
//     MemMan(new ClingMemoryManager(*this));
//   builder.setMCJITMemoryManager(std::move(MemMan));

//   // EngineBuilder uses default c'ted TargetOptions, too:
//   llvm::TargetOptions TargetOpts;
//   TargetOpts.NoFramePointerElim = 1;
//   TargetOpts.JITEmitDebugInfo = 1;

//   builder.setTargetOptions(TargetOpts);

//   m_engine.reset(builder.create());
//   assert(m_engine && "Cannot create module!");
// #endif
}


llvm::JITSymbol
IncrementalJIT::getInjectedSymbols(const std::string& Name) const {
  using JITSymbol = llvm::JITSymbol;
  auto SymMapI = m_SymbolMap.find(Name);
  if (SymMapI != m_SymbolMap.end())
    return JITSymbol(SymMapI->second, llvm::JITSymbolFlags::Exported);

  return JITSymbol(nullptr);
}

std::pair<void*, bool>
IncrementalJIT::lookupSymbol(llvm::StringRef Name, void *InAddr, bool Jit) {
  // FIXME: See comments on DLSym below.
#if !defined(LLVM_ON_WIN32)
  void* Addr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(Name);
#else
  void* Addr = const_cast<void*>(platform::DLSym(Name));
#endif

  if (InAddr && (!Addr || Jit)) {
    if (Jit) {
      std::string Key(Name);
#ifdef MANGLE_PREFIX
      Key.insert(0, MANGLE_PREFIX);
#endif
      m_SymbolMap[Key] = llvm::JITTargetAddress(InAddr);
    }
    llvm::sys::DynamicLibrary::AddSymbol(Name, InAddr);
    return std::make_pair(InAddr, true);
  }
  return std::make_pair(Addr, false);
}
    
llvm::JITSymbol
IncrementalJIT::getSymbolAddressWithoutMangling(const std::string& Name,
                                                bool AlsoInProcess) {
  if (auto Sym = getInjectedSymbols(Name))
    return Sym;

  if (AlsoInProcess) {
    if (llvm::JITSymbol SymInfo = m_ExeMM->findSymbol(Name)) {
      if (auto AddrOrErr = SymInfo.getAddress())
        return llvm::JITSymbol(*AddrOrErr, llvm::JITSymbolFlags::Exported);
      else
        llvm_unreachable("Handle the error case");
    }
#ifdef LLVM_ON_WIN32
    // FIXME: DLSym symbol lookup can overlap m_ExeMM->findSymbol wasting time
    // looking for a symbol in libs where it is already known not to exist.
    // Perhaps a better solution would be to have IncrementalJIT own the
    // DynamicLibraryManger instance (or at least have a reference) that will
    // look only through user loaded libraries.
    // An upside to doing it this way is RTLD_GLOBAL won't need to be used
    // allowing libs with competing symbols to co-exists.
    if (const void* Sym = platform::DLSym(Name))
      return llvm::JITSymbol(llvm::JITTargetAddress(Sym),
                             llvm::JITSymbolFlags::Exported);
#endif
  }

  if (auto Sym = m_LazyEmitLayer.findSymbol(Name, false))
    return Sym;

  return llvm::JITSymbol(nullptr);
}

void IncrementalJIT::addModule(const std::shared_ptr<llvm::Module>& module) {
  // If this module doesn't have a DataLayout attached then attach the
  // default.
  module->setDataLayout(m_TMDataLayout);

  // LLVM MERGE FIXME: update this to use new interfaces.
  auto Resolver = llvm::orc::createLambdaResolver(
    [&](const std::string &S) {
      if (auto Sym = getInjectedSymbols(S)) {
        if (auto AddrOrErr = Sym.getAddress())
          return JITSymbol((uint64_t)*AddrOrErr, Sym.getFlags());
        else
          llvm_unreachable("Handle the error case");
      }
      return m_ExeMM->findSymbol(S);
    },
    [&](const std::string &Name) {
      if (auto Sym = getSymbolAddressWithoutMangling(Name, true)) {
        if (auto AddrOrErr = Sym.getAddress())
          return JITSymbol(*AddrOrErr, Sym.getFlags());
        else
          llvm_unreachable("Handle the error case");
        }

      const std::string* NameNP = &Name;
#ifdef MANGLE_PREFIX
      std::string NameNoPrefix;
      const size_t PrfxLen = strlen(MANGLE_PREFIX);
      if (!Name.compare(0, PrfxLen, MANGLE_PREFIX)) {
        NameNoPrefix = Name.substr(PrfxLen);
        NameNP = &NameNoPrefix;
      }
#endif

      /// This method returns the address of the specified function or variable
      /// that could not be resolved by getSymbolAddress() or by resolving
      /// possible weak symbols by the ExecutionEngine.
      /// It is used to resolve symbols during module linking.

      uint64_t addr = uint64_t(getParent().NotifyLazyFunctionCreators(*NameNP));
      return JITSymbol(addr, llvm::JITSymbolFlags::Weak);
    });

  if (auto H = m_LazyEmitLayer.addModule(module, std::move(Resolver)))
    m_UnloadPoints[module.get()] = *H;
  else
    llvm_unreachable("Handle the error case");
}

llvm::Error
IncrementalJIT::removeModule(const std::shared_ptr<llvm::Module>& module) {
  // FIXME: Track down what calls this routine on a not-yet-added module. Once
  // this is resolved we can remove this check enabling the assert.
  auto IUnload = m_UnloadPoints.find(module.get());
  if (IUnload == m_UnloadPoints.end())
    return llvm::Error::success();
  auto Handle = IUnload->second;
  assert(*Handle && "Trying to remove a non existent module!");
  m_UnloadPoints.erase(IUnload);
  return m_LazyEmitLayer.removeModule(Handle);
}

}// end namespace cling
