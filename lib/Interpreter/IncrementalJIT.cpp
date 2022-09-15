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

#include <sstream>

using namespace llvm;
using namespace llvm::orc;

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

  // FIXME: In LLVM 13 this only works for ELF and MachO platforms
  // Builder.setObjectLinkingLayerCreator(
  //     [&](ExecutionSession &ES, const Triple &TT) {
  //       return std::make_unique<ObjectLinkingLayer>(
  //           ES, std::make_unique<jitlink::InProcessMemoryManager>());
  //     });

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

JITTargetAddress IncrementalJIT::addOrReplaceDefinition(StringRef LinkerMangledName,
                                                        JITTargetAddress KnownAddr) {
  void* Symbol = getSymbolAddress(LinkerMangledName, /*IncludeFromHost=*/true);

  // Nothing to define, we are redefining the same function. FIXME: Diagnose.
  if (Symbol && (JITTargetAddress)Symbol == KnownAddr)
    return KnownAddr;

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
