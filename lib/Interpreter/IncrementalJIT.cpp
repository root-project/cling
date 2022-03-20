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

#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>

using namespace llvm;
using namespace llvm::orc;

namespace cling {

/// Allows to skip the wrapped definition generator based on a shared flag in
/// a single-threaded context.
class SkippableDefinitionGeneratorWrapper : public DefinitionGenerator {
public:
  SkippableDefinitionGeneratorWrapper(
      std::unique_ptr<DefinitionGenerator> DefGenerator,
      SharedAtomicFlag SkipFlag)
      : DefGenerator(std::move(DefGenerator)), SkipFlag(std::move(SkipFlag)) {}

  Error tryToGenerate(LookupState& LS, LookupKind K, JITDylib& JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet& LookupSet) override {
    if (SkipFlag)
      return Error::success();
    return DefGenerator->tryToGenerate(LS, K, JD, JDLookupFlags, LookupSet);
  }

private:
  std::unique_ptr<DefinitionGenerator> DefGenerator;
  SharedAtomicFlag SkipFlag;
};

class NotifyLazyFunctionCreatorsGenerator : public DefinitionGenerator {
  const IncrementalExecutor & m_IncrExecutor;
public:
  NotifyLazyFunctionCreatorsGenerator(const IncrementalExecutor &Exe)
    : m_IncrExecutor(Exe) { }

  Error tryToGenerate(LookupState& LS, LookupKind K, JITDylib& JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet& LookupSet) override {
    SymbolNameSet Missing;
    for (llvm::orc::SymbolStringPtr Name : LookupSet.getSymbolNames())
      if (!m_IncrExecutor.NotifyLazyFunctionCreators((*Name).str()))
        Missing.insert(Name);

    if (!Missing.empty())
      return make_error<SymbolsNotFound>(std::move(Missing));
    return llvm::Error::success();
  }
};

IncrementalJIT::IncrementalJIT(
    IncrementalExecutor& Executor, std::unique_ptr<TargetMachine> TM,
    std::unique_ptr<llvm::orc::ExecutorProcessControl> EPC, Error& Err)
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

  // FIXME: In LLVM 13 this only works for ELF and MachO platforms
  Builder.setObjectLinkingLayerCreator(
      [&](ExecutionSession &ES, const Triple &TT) {
        return std::make_unique<ObjectLinkingLayer>(
            ES, std::make_unique<jitlink::InProcessMemoryManager>());
      });

  if (Expected<std::unique_ptr<LLJIT>> JitInstance = Builder.create()) {
    Jit = std::move(*JitInstance);
  } else {
    Err = JitInstance.takeError();
    return;
  }

  // We use this callback to transfer the ownership of the ThreadSafeModule,
  // which owns the Transaction's llvm::Module, to m_CompiledModules.
  Jit->getIRCompileLayer().setNotifyCompiled([this](auto &MR, ThreadSafeModule TSM) {
      // FIXME: Don't store them mapped by raw pointers.
      const Module *Unsafe = TSM.getModuleUnlocked();
      assert(!m_CompiledModules.count(Unsafe) && "Modules are compiled once");
      m_CompiledModules[Unsafe] = std::move(TSM);
    });

  // FIXME: Make host process symbol lookup optional on a per-query basis
  char LinkerPrefix = this->TM->createDataLayout().getGlobalPrefix();
  Expected<std::unique_ptr<DynamicLibrarySearchGenerator>> HostProcessLookup =
      DynamicLibrarySearchGenerator::GetForCurrentProcess(LinkerPrefix);
  if (!HostProcessLookup) {
    Err = HostProcessLookup.takeError();
    return;
  }

  // Jit->getMainJITDylib().addGenerator(
  //     std::make_unique<SkippableDefinitionGeneratorWrapper>(
  //         std::move(*HostProcessLookup), SkipHostProcessLookup));
  Jit->getMainJITDylib().addGenerator(std::move(*HostProcessLookup));
  auto Notifier = std::make_unique<NotifyLazyFunctionCreatorsGenerator>(Executor);
  Jit->getMainJITDylib().addGenerator(std::move(Notifier));
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
    return std::move(Err);
  return llvm::Error::success();
}

JITTargetAddress IncrementalJIT::addDefinition(StringRef LinkerMangledName,
                                               JITTargetAddress KnownAddr,
                                               bool AcceptExisting) {
  Expected<JITEvaluatedSymbol> Symbol =
      Jit->lookupLinkerMangled(LinkerMangledName);
  if (!Symbol && !KnownAddr) {
    logAllUnhandledErrors(Symbol.takeError(), errs(),
                          "[IncrementalJIT] lookup failed: ");
    return JITTargetAddress{};
  }

  if (KnownAddr && Symbol && !AcceptExisting) {
    errs() << "[IncrementalJIT] cannot redefine existing symbol"
            << " '" << LinkerMangledName << "'\n";
    return JITTargetAddress{};
  }

  if (Symbol)
    return Symbol->getAddress();

  // Let's inject it
  consumeError(Symbol.takeError());

  bool Inserted;
  SymbolMap::iterator It;
  std::tie(It, Inserted) = InjectedSymbols.try_emplace(
      Jit->getExecutionSession().intern(LinkerMangledName),
      JITEvaluatedSymbol(KnownAddr, JITSymbolFlags::Exported));
  assert(Inserted && "Why wasn't this found in the initial Jit lookup?");

  if (Error Err = Jit->getMainJITDylib().define(absoluteSymbols({*It}))) {
    logAllUnhandledErrors(std::move(Err), errs(),
                          "[IncrementalJIT] define() failed: ");
    return JITTargetAddress{};
  }

  return KnownAddr;
}

void* IncrementalJIT::getSymbolAddress(StringRef Name, bool ExcludeHostSymbols) {
  std::unique_lock<SharedAtomicFlag> G(SkipHostProcessLookup, std::defer_lock);
  if (ExcludeHostSymbols)
    G.lock();

  Expected<JITEvaluatedSymbol> Symbol = Jit->lookup(Name);
  if (!Symbol) {
    logAllUnhandledErrors(Symbol.takeError(), errs(),
                          "[IncrementalJIT] getSymbolAddress() failed: ");
    return 0;
  }

  return jitTargetAddressToPointer<void*>(Symbol->getAddress());
}

} // namespace cling
