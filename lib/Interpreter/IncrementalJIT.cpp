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
  cling::IncrementalExecutor& m_exe;

public:
  ClingMemoryManager(cling::IncrementalExecutor& Exe):
    m_exe(Exe) {}

  /// This method returns the address of the specified function or variable
  /// that could not be resolved by getSymbolAddress() or by resolving
  /// possible weak symbols by the ExecutionEngine.
  /// It is used to resolve symbols during module linking.
  uint64_t getMissingSymbolAddress(const std::string &Name) override {
    return (uint64_t) m_exe.NotifyLazyFunctionCreators(Name);
  }

  ///\brief Simply wraps the base class's function setting AbortOnFailure
  /// to false and instead using the error handling mechanism to report it.
  void* getPointerToNamedFunction(const std::string &Name,
                                  bool /*AbortOnFailure*/ =true) override {
    return SectionMemoryManager::getPointerToNamedFunction(Name, false);
  }
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

  void reserveAllocationSpace(uintptr_t CodeSize, uintptr_t DataSizeRO,
                              uintptr_t DataSizeRW) override {
    return getExeMM()->reserveAllocationSpace(CodeSize, DataSizeRO, DataSizeRW);
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
    return m_jit.getSymbolAddressWithoutMangling(Name);
  }

  void *getPointerToNamedFunction(const std::string &Name,
                                  bool AbortOnFailure = true) override {
    return getExeMM()->getPointerToNamedFunction(Name, AbortOnFailure);
  }

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

  /// This method returns the address of the specified function or variable
  /// that could not be resolved by getSymbolAddress() or by resolving
  /// possible weak symbols by the ExecutionEngine.
  /// It is used to resolve symbols during module linking.
  uint64_t getMissingSymbolAddress(const std::string &Name) override {
    std::string NameNoPrefix;
    if (MANGLE_PREFIX[0]
        && !Name.compare(0, strlen(MANGLE_PREFIX), MANGLE_PREFIX))
      NameNoPrefix = Name.substr(strlen(MANGLE_PREFIX), -1);
    else
      NameNoPrefix = std::move(Name);
    return (uint64_t) m_jit.getParent().NotifyLazyFunctionCreators(NameNoPrefix);
  }


}; // class Azog

IncrementalJIT::IncrementalJIT(IncrementalExecutor& exe,
                               std::unique_ptr<TargetMachine> TM):
  m_Parent(exe),
  m_TM(std::move(TM)),
  m_ExeMM(std::move(llvm::make_unique<ClingMemoryManager>(m_Parent))),
  m_Mang(m_TM->getDataLayout()),
  m_NotifyObjectLoaded(*this), m_NotifyFinalized(*this),
  m_ObjectLayer(ObjectLayerT::CreateRTDyldMMFtor(), m_NotifyObjectLoaded,
                m_NotifyFinalized),
  m_CompileLayer(m_ObjectLayer, SimpleCompiler(*m_TM)),
  m_LazyEmitLayer(m_CompileLayer) {

  // Enable JIT symbol resolution from the binary.
  llvm::sys::DynamicLibrary::LoadLibraryPermanently(0, 0);

  // Make debug symbols available.
  m_GDBListener = JITEventListener::createGDBRegistrationListener();

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


uint64_t IncrementalJIT::getSymbolAddressWithoutMangling(llvm::StringRef Name) {
  if (Name == MANGLE_PREFIX "__cxa_atexit") {
    // Rewire __cxa_atexit to ~Interpreter(), thus also global destruction
    // coming from the JIT.
    return (uint64_t)&local_cxa_atexit;
  } else if (Name == MANGLE_PREFIX "__dso_handle") {
    // Provide IncrementalExecutor as the third argument to __cxa_atexit.
    return (uint64_t)&m_Parent;
  }
  if (uint64_t Addr = m_ExeMM->getSymbolAddress(Name))
    return Addr;
  if (uint64_t Addr = m_LazyEmitLayer.getSymbolAddress(Name, false))
    return Addr;

  return 0;
}

size_t IncrementalJIT::addModules(std::vector<llvm::Module*>&& modules) {
  // If this module doesn't have a DataLayout attached then attach the
  // default.
  for (auto&& mod: modules) {
    if (!mod->getDataLayout())
      mod->setDataLayout(m_TM->getDataLayout());
  }

  ModuleSetHandleT MSHandle
    = m_LazyEmitLayer.addModuleSet(std::move(modules),
                                   llvm::make_unique<Azog>(*this));
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
  m_LazyEmitLayer.removeModuleSet(m_UnloadPoints[handle]);
}

}// end namespace cling
