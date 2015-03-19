//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_INCREMENTAL_JIT_H
#define CLING_INCREMENTAL_JIT_H

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "llvm/IR/Mangler.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LazyEmittingLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class Module;
  class RTDyldMemoryManager;
}

namespace cling {
class Azog;
class IncrementalExecutor;

class IncrementalJIT {
  friend class Azog;

  ///\brief The IncrementalExecutor who owns us.
  IncrementalExecutor& m_Parent;
  llvm::JITEventListener* m_GDBListener; // owned by llvm::ManagedStaticBase

  class NotifyObjectLoadedT {
  public:
    typedef std::vector<std::unique_ptr<llvm::object::ObjectFile>> ObjListT;
    typedef std::vector<std::unique_ptr<llvm::RuntimeDyld::LoadedObjectInfo>>
        LoadedObjInfoListT;

    NotifyObjectLoadedT(IncrementalJIT &jit) : m_JIT(jit) {}

    void operator()(llvm::ObjectLinkingLayerBase::ObjSetHandleT H,
                    const ObjListT &Objects,
                    const LoadedObjInfoListT &Infos) const {
      m_JIT.m_UnfinalizedSections[H]
        = std::move(m_JIT.m_SectionsAllocatedSinceLastLoad);
      m_JIT.m_SectionsAllocatedSinceLastLoad = SectionAddrSet();
      assert(Objects.size() == Infos.size() &&
             "Incorrect number of Infos for Objects.");
      for (size_t I = 0, N = Objects.size(); I < N; ++I)
        m_JIT.m_GDBListener->NotifyObjectEmitted(*Objects[I], *Infos[I]);
    };

  private:
    IncrementalJIT &m_JIT;
  };

  class NotifyFinalizedT {
  public:
    NotifyFinalizedT(IncrementalJIT &jit) : m_JIT(jit) {}
    void operator()(llvm::ObjectLinkingLayerBase::ObjSetHandleT H) {
      m_JIT.m_UnfinalizedSections.erase(H);
    }

  private:
    IncrementalJIT &m_JIT;
  };


  typedef llvm::ObjectLinkingLayer<NotifyObjectLoadedT> ObjectLayerT;
  typedef llvm::IRCompileLayer<ObjectLayerT> CompileLayerT;
  typedef llvm::LazyEmittingLayer<CompileLayerT> LazyEmitLayerT;
  typedef LazyEmitLayerT::ModuleSetHandleT ModuleSetHandleT;

  std::unique_ptr<llvm::TargetMachine> m_TM;
  ///\brief The RTDyldMemoryManager used to communicate with the
  /// IncrementalExecutor to handle missing or special symbols.
  std::unique_ptr<llvm::RTDyldMemoryManager> m_ExeMM;

  ///\brief Target symbol mangler.
  llvm::Mangler m_Mang;

  NotifyObjectLoadedT m_NotifyObjectLoaded;
  NotifyFinalizedT m_NotifyFinalized;

  ObjectLayerT m_ObjectLayer;
  CompileLayerT m_CompileLayer;
  LazyEmitLayerT m_LazyEmitLayer;

  // We need to store ObjLayerT::ObjSetHandles for each of the object sets
  // that have been emitted but not yet finalized so that we can forward the
  // mapSectionAddress calls appropriately.
  typedef std::set<const void *> SectionAddrSet;
  struct ObjSetHandleCompare {
    bool operator()(ObjectLayerT::ObjSetHandleT H1,
                    ObjectLayerT::ObjSetHandleT H2) const {
      return &*H1 < &*H2;
    }
  };
  SectionAddrSet m_SectionsAllocatedSinceLastLoad;
  std::map<ObjectLayerT::ObjSetHandleT, SectionAddrSet, ObjSetHandleCompare>
      m_UnfinalizedSections;

  ///\brief Vector of ModuleSetHandleT. UnloadHandles index into that
  /// vector.
  std::vector<ModuleSetHandleT> m_UnloadPoints;


  std::string Mangle(llvm::StringRef Name) {
    std::string MangledName;
    {
      llvm::raw_string_ostream MangledNameStream(MangledName);
      m_Mang.getNameWithPrefix(MangledNameStream, Name);
    }
    return MangledName;
  }

public:
  IncrementalJIT(IncrementalExecutor& exe,
                 std::unique_ptr<llvm::TargetMachine> TM);

  ///\brief Get the address of a symbol from the JIT or the memory manager,
  /// mangling the name as needed. Use this to resolve symbols as coming
  /// from clang's mangler.
  uint64_t getSymbolAddress(llvm::StringRef Name) {
    return getSymbolAddressWithoutMangling(Mangle(Name));
  }

  ///\brief Get the address of a symbol from the JIT or the memory manager.
  /// Use this to resolve symbols of known, target-specific names.
  uint64_t getSymbolAddressWithoutMangling(llvm::StringRef Name);

  size_t addModules(std::vector<llvm::Module*>&& modules);
  void removeModules(size_t handle);

  IncrementalExecutor& getParent() const { return m_Parent; }
  //void finalizeMemory();
};
} // end cling
#endif // CLING_INCREMENTAL_EXECUTOR_H
