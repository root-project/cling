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

#include "cling/Utils/Output.h"

#include "llvm/IR/Mangler.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LazyEmittingLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/Target/TargetMachine.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace llvm {
class Module;
class RTDyldMemoryManager;
}

namespace cling {
class Azog;
class IncrementalExecutor;

class IncrementalJIT {
public:
  using SymbolMapT = llvm::StringMap<llvm::JITTargetAddress>;

private:
  friend class Azog;

  ///\brief The IncrementalExecutor who owns us.
  IncrementalExecutor& m_Parent;
  llvm::JITEventListener* m_GDBListener; // owned by llvm::ManagedStaticBase

  SymbolMapT m_SymbolMap;

  class NotifyObjectLoadedT {
  public:
    NotifyObjectLoadedT(IncrementalJIT &jit) : m_JIT(jit) {}
    void operator()(llvm::orc::RTDyldObjectLinkingLayerBase::ObjHandleT H,
                    const llvm::orc::RTDyldObjectLinkingLayer::ObjectPtr &Object,
                    const llvm::LoadedObjectInfo &Info) const {
      m_JIT.m_UnfinalizedSections[H]
        = std::move(m_JIT.m_SectionsAllocatedSinceLastLoad);
      m_JIT.m_SectionsAllocatedSinceLastLoad = SectionAddrSet();

      // FIXME: NotifyObjectEmitted requires a RuntimeDyld::LoadedObjectInfo
      // object. In order to get it one should call
      // RTDyld.loadObject(*ObjToLoad->getBinary()) according to r306058.
      // Moreover this should be done in the finalizer. Currently we are
      // disabling this since we have globally disabled this functionality in
      // IncrementalJIT.cpp (m_GDBListener = 0).
      //
      // if (auto GDBListener = m_JIT.m_GDBListener)
      //   GDBListener->NotifyObjectEmitted(*Object->getBinary(), Info);

      for (const auto &Symbol: Object->getBinary()->symbols()) {
        auto Flags = Symbol.getFlags();
        if (Flags & llvm::object::BasicSymbolRef::SF_Undefined)
          continue;
        // FIXME: this should be uncommented once we serve incremental
        // modules from a TU module.
        //if (!(Flags & llvm::object::BasicSymbolRef::SF_Exported))
        //  continue;
        auto NameOrError = Symbol.getName();
        if (!NameOrError)
          continue;
        auto Name = NameOrError.get();
        if (m_JIT.m_SymbolMap.find(Name) == m_JIT.m_SymbolMap.end()) {
          llvm::JITSymbol Sym
            = m_JIT.m_CompileLayer.findSymbolIn(H, Name, true);
          if (auto Addr = Sym.getAddress())
            m_JIT.m_SymbolMap[Name] = *Addr;
        }
      }
    }

  private:
    IncrementalJIT &m_JIT;
  };
  class RemovableObjectLinkingLayer:
    public llvm::orc::RTDyldObjectLinkingLayer {
  public:
    using Base_t = llvm::orc::RTDyldObjectLinkingLayer;
    using NotifyFinalizedFtor = Base_t::NotifyFinalizedFtor;
    RemovableObjectLinkingLayer(SymbolMapT &SymMap,
                                Base_t::MemoryManagerGetter MM,
                                NotifyObjectLoadedT NotifyLoaded,
                                NotifyFinalizedFtor NotifyFinalized)
      : Base_t(MM, NotifyLoaded, NotifyFinalized), m_SymbolMap(SymMap)
    {}

    llvm::Error
    removeObject(llvm::orc::RTDyldObjectLinkingLayerBase::ObjHandleT H) {
      struct AccessSymbolTable: public LinkedObject {
        const llvm::StringMap<llvm::JITEvaluatedSymbol>&
        getSymbolTable() const {
          return SymbolTable;
        }
      };
      const AccessSymbolTable* HSymTable
        = static_cast<const AccessSymbolTable*>(H->get());
      for (auto&& NameSym: HSymTable->getSymbolTable()) {
        auto iterSymMap = m_SymbolMap.find(NameSym.first());
        if (iterSymMap == m_SymbolMap.end())
          continue;
        // Is this this symbol (address)?
        if (iterSymMap->second == NameSym.second.getAddress())
          m_SymbolMap.erase(iterSymMap);
      }
      return llvm::orc::RTDyldObjectLinkingLayer::removeObject(H);
    }
  private:
    SymbolMapT& m_SymbolMap;
  };

  typedef RemovableObjectLinkingLayer ObjectLayerT;
  typedef llvm::orc::IRCompileLayer<ObjectLayerT,
                                    llvm::orc::SimpleCompiler> CompileLayerT;
  typedef llvm::orc::LazyEmittingLayer<CompileLayerT> LazyEmitLayerT;
  typedef LazyEmitLayerT::ModuleHandleT ModuleHandleT;

  std::unique_ptr<llvm::TargetMachine> m_TM;
  llvm::DataLayout m_TMDataLayout;

  ///\brief The RTDyldMemoryManager used to communicate with the
  /// IncrementalExecutor to handle missing or special symbols.
  std::shared_ptr<llvm::RTDyldMemoryManager> m_ExeMM;

  NotifyObjectLoadedT m_NotifyObjectLoaded;

  ObjectLayerT m_ObjectLayer;
  CompileLayerT m_CompileLayer;
  LazyEmitLayerT m_LazyEmitLayer;

  // We need to store ObjLayerT::ObjHandles for each of the object sets
  // that have been emitted but not yet finalized so that we can forward the
  // mapSectionAddress calls appropriately.
  typedef std::set<const void *> SectionAddrSet;
  struct ObjHandleCompare {
    bool operator()(ObjectLayerT::ObjHandleT H1,
                    ObjectLayerT::ObjHandleT H2) const {
      return &*H1 < &*H2;
    }
  };
  SectionAddrSet m_SectionsAllocatedSinceLastLoad;
  std::map<ObjectLayerT::ObjHandleT, SectionAddrSet, ObjHandleCompare>
    m_UnfinalizedSections;

  ///\brief Mapping between \c llvm::Module* and \c ModuleHandleT.
  std::map<llvm::Module*, ModuleHandleT> m_UnloadPoints;

  std::string Mangle(llvm::StringRef Name) {
    stdstrstream MangledName;
    llvm::Mangler::getNameWithPrefix(MangledName, Name, m_TMDataLayout);
    return MangledName.str();
  }

  llvm::JITSymbol getInjectedSymbols(const std::string& Name) const;

public:
  IncrementalJIT(IncrementalExecutor& exe,
                 std::unique_ptr<llvm::TargetMachine> TM);

  ///\brief Get the address of a symbol from the JIT or the memory manager,
  /// mangling the name as needed. Use this to resolve symbols as coming
  /// from clang's mangler.
  /// \param Name - name to look for. This name might still get mangled
  ///   (prefixed by '_') to make IR versus symbol names.
  /// \param AlsoInProcess - Sometimes you only care about JITed symbols. If so,
  ///   pass `false` here to not resolve the symbol through dlsym().
  uint64_t getSymbolAddress(const std::string& Name, bool AlsoInProcess) {
    // FIXME: We should decide if we want to handle the error here or make the
    // return type of the function llvm::Expected<uint64_t> relying on the
    // users to decide how to handle the error.
    if (auto S = getSymbolAddressWithoutMangling(Mangle(Name), AlsoInProcess)) {
      if (auto AddrOrErr = S.getAddress())
        return *AddrOrErr;
      else
        llvm_unreachable("Handle the error case");
    }

    return 0;
  }

  ///\brief Get the address of a symbol from the JIT or the memory manager.
  /// Use this to resolve symbols of known, target-specific names.
  llvm::JITSymbol getSymbolAddressWithoutMangling(const std::string& Name,
                                                  bool AlsoInProcess);

  void addModule(const std::shared_ptr<llvm::Module>& module);
  llvm::Error removeModule(const std::shared_ptr<llvm::Module>& module);

  IncrementalExecutor& getParent() const { return m_Parent; }

  void RemoveUnfinalizedSection(
                     llvm::orc::RTDyldObjectLinkingLayerBase::ObjHandleT H) {
    m_UnfinalizedSections.erase(H);
  }

  ///\brief Get the address of a symbol from the process' loaded libraries.
  /// \param Name - symbol to look for
  /// \param Addr - known address of the symbol that can be cached later use
  /// \param Jit - add to the injected symbols cache
  /// \returns The address of the symbol and whether it was cached
  std::pair<void*, bool>
  lookupSymbol(llvm::StringRef Name, void* Addr = nullptr, bool Jit = false);
};
} // end cling
#endif // CLING_INCREMENTAL_EXECUTOR_H
