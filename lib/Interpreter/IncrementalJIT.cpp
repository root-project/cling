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

#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/Support/DynamicLibrary.h"

#ifdef __APPLE__
// Apple adds an extra '_'
# define MANGLE_PREFIX "_"
#else
# define MANGLE_PREFIX ""
#endif

using namespace llvm;

namespace {
// Forward cxa_atexit for global d'tors.
static void local_cxa_atexit(void (*func) (void*), void* arg, void* dso) {
  cling::IncrementalExecutor* exe = (cling::IncrementalExecutor*)dso;
  exe->AddAtExitFunc(func, arg);
}

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
    void operator()(llvm::orc::ObjectLinkingLayerBase::ObjSetHandleT H) {
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

public:
  Azog(cling::IncrementalJIT& Jit): m_jit(Jit) {}

  RTDyldMemoryManager* getExeMM() const { return m_jit.m_ExeMM.get(); }

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override {
    uint8_t *Addr =
      getExeMM()->allocateCodeSection(Size, Alignment, SectionID, SectionName);
    m_jit.m_SectionsAllocatedSinceLastLoad.insert(Addr);
    return Addr;
  }

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool IsReadOnly) override {
    uint8_t *Addr = getExeMM()->allocateDataSection(Size, Alignment, SectionID,
                                                    SectionName, IsReadOnly);
    m_jit.m_SectionsAllocatedSinceLastLoad.insert(Addr);
    return Addr;
  }

  void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                              uintptr_t RODataSize, uint32_t RODataAlign,
                              uintptr_t RWDataSize, uint32_t RWDataAlign) override {
    return getExeMM()->reserveAllocationSpace(CodeSize, CodeAlign, RODataSize,
                                              RODataAlign, RWDataSize,
                                              RWDataAlign);
  }

  bool needsToReserveAllocationSpace() override {
    return getExeMM()->needsToReserveAllocationSpace();
  }

  void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                        size_t Size) override {
    return getExeMM()->registerEHFrames(Addr, LoadAddr, Size);
  }

  void deregisterEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                          size_t Size) override {
    return getExeMM()->deregisterEHFrames(Addr, LoadAddr, Size);
  }

  uint64_t getSymbolAddress(const std::string &Name) override {
    return m_jit.getSymbolAddressWithoutMangling(Name,
                                                 true /*also use dlsym*/)
      .getAddress();
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
    if (m_jit.m_UnfinalizedSections.size() == 1)
      return getExeMM()->finalizeMemory(ErrMsg);
    return false;
  };

}; // class Azog

IncrementalJIT::IncrementalJIT(IncrementalExecutor& exe,
                               std::unique_ptr<TargetMachine> TM):
  m_Parent(exe),
  m_TM(std::move(TM)),
  m_TMDataLayout(m_TM->createDataLayout()),
  m_ExeMM(llvm::make_unique<ClingMemoryManager>(m_Parent)),
  m_NotifyObjectLoaded(*this),
  m_ObjectLayer(m_SymbolMap, m_NotifyObjectLoaded, NotifyFinalizedT(*this)),
  m_CompileLayer(m_ObjectLayer, llvm::orc::SimpleCompiler(*m_TM)),
  m_LazyEmitLayer(m_CompileLayer) {

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


llvm::orc::JITSymbol
IncrementalJIT::getInjectedSymbols(llvm::StringRef Name) const {
  using JITSymbol = llvm::orc::JITSymbol;
  if (Name == MANGLE_PREFIX "__cxa_atexit") {
    // Rewire __cxa_atexit to ~Interpreter(), thus also global destruction
    // coming from the JIT.
    return JITSymbol((uint64_t)&local_cxa_atexit,
                     llvm::JITSymbolFlags::Exported);
  } else if (Name == MANGLE_PREFIX "__dso_handle") {
    // Provide IncrementalExecutor as the third argument to __cxa_atexit.
    return JITSymbol((uint64_t)&m_Parent,
                     llvm::JITSymbolFlags::Exported);
  }

  auto SymMapI = m_SymbolMap.find(Name);
  if (SymMapI != m_SymbolMap.end())
    return JITSymbol(SymMapI->second, llvm::JITSymbolFlags::Exported);

  return JITSymbol(nullptr);
}

llvm::orc::JITSymbol
IncrementalJIT::getSymbolAddressWithoutMangling(llvm::StringRef Name,
                                                bool AlsoInProcess) {
  if (auto Sym = getInjectedSymbols(Name))
    return Sym;

  if (AlsoInProcess) {
    if (RuntimeDyld::SymbolInfo SymInfo = m_ExeMM->findSymbol(Name))
      return llvm::orc::JITSymbol(SymInfo.getAddress(),
                                  llvm::JITSymbolFlags::Exported);
  }

  if (auto Sym = m_LazyEmitLayer.findSymbol(Name, false))
    return Sym;

  return llvm::orc::JITSymbol(nullptr);
}

size_t IncrementalJIT::addModules(std::vector<llvm::Module*>&& modules) {
  // If this module doesn't have a DataLayout attached then attach the
  // default.
  for (auto&& mod: modules) {
    mod->setDataLayout(m_TMDataLayout);
  }

  // LLVM MERGE FIXME: update this to use new interfaces.
  auto Resolver = llvm::orc::createLambdaResolver(
    [&](const std::string &S) {
      if (auto Sym = getInjectedSymbols(S))
        return RuntimeDyld::SymbolInfo((uint64_t)Sym.getAddress(),
                                       Sym.getFlags());
      return m_ExeMM->findSymbol(S);
    },
    [&](const std::string &Name) {
      if (auto Sym = getSymbolAddressWithoutMangling(Name, true)
          /*was: findSymbol(Name)*/)
        return RuntimeDyld::SymbolInfo(Sym.getAddress(),
                                       Sym.getFlags());


      /// This method returns the address of the specified function or variable
      /// that could not be resolved by getSymbolAddress() or by resolving
      /// possible weak symbols by the ExecutionEngine.
      /// It is used to resolve symbols during module linking.

      std::string NameNoPrefix;
      if (MANGLE_PREFIX[0]
          && !Name.compare(0, strlen(MANGLE_PREFIX), MANGLE_PREFIX))
        NameNoPrefix = Name.substr(strlen(MANGLE_PREFIX), -1);
      else
        NameNoPrefix = std::move(Name);
      uint64_t addr
        = (uint64_t) getParent().NotifyLazyFunctionCreators(NameNoPrefix);
      return RuntimeDyld::SymbolInfo(addr, llvm::JITSymbolFlags::Weak);
    });

  ModuleSetHandleT MSHandle
    = m_LazyEmitLayer.addModuleSet(std::move(modules),
                                   llvm::make_unique<Azog>(*this),
                                   std::move(Resolver));
  m_UnloadPoints.push_back(MSHandle);
  return m_UnloadPoints.size() - 1;
}

// void* IncrementalJIT::finalizeMemory() {
//   for (auto &P : UnfinalizedSections)
//     if (P.second.count(LocalAddress))
//       ObjectLayer.mapSectionAddress(P.first, LocalAddress, TargetAddress);
// }


void IncrementalJIT::removeModules(size_t handle) {
  if (handle == (size_t)-1)
    return;
  auto objSetHandle = m_UnloadPoints[handle];
  m_LazyEmitLayer.removeModuleSet(objSetHandle);
}

}// end namespace cling
