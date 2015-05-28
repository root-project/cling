//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IncrementalExecutor.h"
#include "IncrementalJIT.h"
#include "Threading.h"

#include "cling/Interpreter/Value.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/Basic/Diagnostic.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/PassManager.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#ifdef LLVM_ON_WIN32
extern "C"
char *__unDName(char *demangled, const char *mangled, int out_len,
                void * (* pAlloc )(size_t), void (* pFree )(void *),
                unsigned short int flags);
#else
#include <cxxabi.h>
#endif

using namespace llvm;

namespace cling {

IncrementalExecutor::IncrementalExecutor(clang::DiagnosticsEngine& diags):
  m_CurrentAtExitModule(0)
#if 0
  : m_Diags(diags)
#endif
{

  // MSVC doesn't support m_AtExitFuncsSpinLock=ATOMIC_FLAG_INIT; in the class definition
  std::atomic_flag_clear( &m_AtExitFuncsSpinLock );

  // No need to protect this access of m_AtExitFuncs, since nobody
  // can use this object yet.
  m_AtExitFuncs.reserve(256);

  m_JIT.reset(new IncrementalJIT(*this, std::move(CreateHostTargetMachine())));
}

// Keep in source: ~unique_ptr<ClingJIT> needs ClingJIT
IncrementalExecutor::~IncrementalExecutor() {}

std::unique_ptr<TargetMachine>
  IncrementalExecutor::CreateHostTargetMachine() const {
  // TODO: make this configurable.
  Triple TheTriple(sys::getProcessTriple());
#ifdef _WIN32
  /*
	* MCJIT works on Windows, but currently only through ELF object format.
	*/
  TheTriple.setObjectFormat(llvm::Triple::ELF);
#endif
  std::string Error;
  const Target *TheTarget
    = TargetRegistry::lookupTarget(TheTriple.getTriple(), Error);
  if (!TheTarget) {
    llvm::errs() << "cling::IncrementalExecutor: unable to find target:\n"
                 << Error;
  }

  std::string MCPU;
  std::string FeaturesStr;

  TargetOptions Options = TargetOptions();
  Options.NoFramePointerElim = 1;
  Options.JITEmitDebugInfo = 1;
  Reloc::Model RelocModel = Reloc::Default;
  CodeModel::Model CMModel = CodeModel::JITDefault;
  CodeGenOpt::Level OptLevel = CodeGenOpt::Less;

  std::unique_ptr<TargetMachine> TM;
  TM.reset(TheTarget->createTargetMachine(TheTriple.getTriple(),
                                          MCPU, FeaturesStr,
                                          Options,
                                          RelocModel, CMModel,
                                          OptLevel));
  return std::move(TM);
}

void IncrementalExecutor::shuttingDown() {
  // No need to protect this access, since hopefully there is no concurrent
  // shutdown request.
  for (size_t I = 0, N = m_AtExitFuncs.size(); I < N; ++I) {
    const CXAAtExitElement& AEE = m_AtExitFuncs[N - I - 1];
    (*AEE.m_Func)(AEE.m_Arg);
  }
}

void IncrementalExecutor::AddAtExitFunc(void (*func) (void*), void* arg) {
  // Register a CXAAtExit function
  cling::internal::SpinLockGuard slg(m_AtExitFuncsSpinLock);
  m_AtExitFuncs.push_back(CXAAtExitElement(func, arg, m_CurrentAtExitModule));
}

void unresolvedSymbol()
{
  // throw exception?
  llvm::errs() << "IncrementalExecutor: calling unresolved symbol, "
    "see previous error message!\n";
}

void* IncrementalExecutor::HandleMissingFunction(const std::string& mangled_name)
{
  // Not found in the map, add the symbol in the list of unresolved symbols
  if (m_unresolvedSymbols.insert(mangled_name).second) {
    //llvm::errs() << "IncrementalExecutor: use of undefined symbol '"
    //             << mangled_name << "'!\n";
  }

  // Avoid "ISO C++ forbids casting between pointer-to-function and
  // pointer-to-object":
  return (void*)reinterpret_cast<size_t>(unresolvedSymbol);
}

void*
IncrementalExecutor::NotifyLazyFunctionCreators(const std::string& mangled_name)
{
  for (std::vector<LazyFunctionCreatorFunc_t>::iterator it
         = m_lazyFuncCreator.begin(), et = m_lazyFuncCreator.end();
       it != et; ++it) {
    void* ret = (void*)((LazyFunctionCreatorFunc_t)*it)(mangled_name);
    if (ret)
      return ret;
  }

  return HandleMissingFunction(mangled_name);
}

#if 0
// FIXME: employ to empty module dependencies *within* the *current* module.
static void
freeCallersOfUnresolvedSymbols(llvm::SmallVectorImpl<llvm::Function*>&
                               funcsToFree, llvm::ExecutionEngine* engine) {
  llvm::SmallPtrSet<llvm::Function*, 40> funcsToFreeUnique;
  for (size_t i = 0; i < funcsToFree.size(); ++i) {
    llvm::Function* func = funcsToFree[i];
    assert(func && "Cannot free NULL function");
    if (funcsToFreeUnique.insert(func).second) {
      for (llvm::Value::use_iterator IU = func->use_begin(),
             EU = func->use_end(); IU != EU; ++IU) {
        llvm::Instruction* instUser = llvm::dyn_cast<llvm::Instruction>(*IU);
        if (!instUser) continue;
        if (!instUser->getParent()) continue;
        if (llvm::Function* userFunc = instUser->getParent()->getParent())
          funcsToFree.push_back(userFunc);
      }
    }
  }
  for (llvm::SmallPtrSet<llvm::Function*, 40>::iterator
         I = funcsToFreeUnique.begin(), E = funcsToFreeUnique.end();
       I != E; ++I) {
    // This should force the JIT to recompile the function. But the stubs stay,
    // and the JIT reuses the stubs now pointing nowhere, i.e. without updating
    // the machine code address. Fix the JIT, or hope that MCJIT helps.
    //engine->freeMachineCodeForFunction(*I);
    engine->updateGlobalMapping(*I, 0);
  }
}
#endif

IncrementalExecutor::ExecutionResult
IncrementalExecutor::runStaticInitializersOnce(const Transaction& T) {
  llvm::Module* m = T.getModule();
  assert(m && "Module must not be null");

  // Set m_CurrentAtExitModule to the Module, unset to 0 once done.
  struct AtExitModuleSetterRAII {
    llvm::Module*& m_AEM;
    AtExitModuleSetterRAII(llvm::Module* M, llvm::Module*& AEM): m_AEM(AEM)
    { AEM = M; }
    ~AtExitModuleSetterRAII() { m_AEM = 0; }
  } DSOHandleSetter(m, m_CurrentAtExitModule);

  // We don't care whether something was unresolved before.
  m_unresolvedSymbols.clear();

  // check if there is any unresolved symbol in the list
  if (diagnoseUnresolvedSymbols("static initializers"))
    return kExeUnresolvedSymbols;

  llvm::GlobalVariable* GV
     = m->getGlobalVariable("llvm.global_ctors", true);
  // Nothing to do is good, too.
  if (!GV) return kExeSuccess;

  // Close similarity to
  // m_engine->runStaticConstructorsDestructors(false) aka
  // llvm::ExecutionEngine::runStaticConstructorsDestructors()
  // is intentional; we do an extra pass to check whether the JIT
  // managed to collect all the symbols needed by the niitializers.
  // Should be an array of '{ i32, void ()* }' structs.  The first value is
  // the init priority, which we ignore.
  llvm::ConstantArray *InitList
    = llvm::dyn_cast<llvm::ConstantArray>(GV->getInitializer());

  // We need to delete it here just in case we have recursive inits, otherwise
  // it will call inits multiple times.
  GV->eraseFromParent();

  if (InitList == 0)
    return kExeSuccess;

  SmallVector<Function*, 2> initFuncs;

  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    llvm::ConstantStruct *CS
      = llvm::dyn_cast<llvm::ConstantStruct>(InitList->getOperand(i));
    if (CS == 0) continue;

    llvm::Constant *FP = CS->getOperand(1);
    if (FP->isNullValue())
      continue;  // Found a sentinal value, ignore.

    // Strip off constant expression casts.
    if (llvm::ConstantExpr *CE = llvm::dyn_cast<llvm::ConstantExpr>(FP))
      if (CE->isCast())
        FP = CE->getOperand(0);

    // Execute the ctor/dtor function!
    if (llvm::Function *F = llvm::dyn_cast<llvm::Function>(FP)) {
      executeInit(F->getName());

      initFuncs.push_back(F);
      if (F->getName().startswith("_GLOBAL__sub_I__")) {
        BasicBlock& BB = F->getEntryBlock();
        for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
          if (CallInst* call = dyn_cast<CallInst>(I))
            initFuncs.push_back(call->getCalledFunction());
      }
    }
  }

  for (SmallVector<Function*,2>::iterator I = initFuncs.begin(),
         E = initFuncs.end(); I != E; ++I) {
    // Cleanup also the dangling init functions. They are in the form:
    // define internal void @_GLOBAL__I_aN() section "..."{
    // entry:
    //   call void @__cxx_global_var_init(N-1)()
    //   call void @__cxx_global_var_initM()
    //   ret void
    // }
    //
    // define internal void @__cxx_global_var_init(N-1)() section "..." {
    // entry:
    //   call void @_ZN7MyClassC1Ev(%struct.MyClass* @n)
    //   ret void
    // }

    // Erase __cxx_global_var_init(N-1)() first.
    (*I)->removeDeadConstantUsers();
    (*I)->eraseFromParent();
  }

  return kExeSuccess;
}

void IncrementalExecutor::runAndRemoveStaticDestructors(Transaction* T) {
  assert(T && "Must be set");
  // Collect all the dtors bound to this transaction.
  AtExitFunctions boundToT;

  {
    cling::internal::SpinLockGuard slg(m_AtExitFuncsSpinLock);
    for (AtExitFunctions::iterator I = m_AtExitFuncs.begin();
         I != m_AtExitFuncs.end();)
      if (I->m_FromM == T->getModule()) {
        boundToT.push_back(*I);
        I = m_AtExitFuncs.erase(I);
      }
      else
        ++I;
  } // end of spin lock lifetime block.

  // 'Unload' the cxa_atexit entities.
  for (AtExitFunctions::reverse_iterator I = boundToT.rbegin(),
         E = boundToT.rend(); I != E; ++I) {
    const CXAAtExitElement& AEE = *I;
    (*AEE.m_Func)(AEE.m_Arg);
  }
}

void
IncrementalExecutor::installLazyFunctionCreator(LazyFunctionCreatorFunc_t fp)
{
  m_lazyFuncCreator.push_back(fp);
}

bool
IncrementalExecutor::addSymbol(const char* symbolName,  void* symbolAddress) {
  void* actualAddress
    = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(symbolName);
  if (actualAddress)
    return false;

  llvm::sys::DynamicLibrary::AddSymbol(symbolName, symbolAddress);
  return true;
}

void* IncrementalExecutor::getAddressOfGlobal(llvm::StringRef symbolName,
                                              bool* fromJIT /*=0*/) {
  // Return a symbol's address, and whether it was jitted.
  void* address
    = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(symbolName);

  // It's not from the JIT if it's in a dylib.
  if (fromJIT)
    *fromJIT = !address;

  if (!address)
    return (void*)m_JIT->getSymbolAddress(symbolName);

  return address;
}

void*
IncrementalExecutor::getPointerToGlobalFromJIT(const llvm::GlobalValue& GV) {
  // Get the function / variable pointer referenced by GV.

  // We don't care whether something was unresolved before.
  m_unresolvedSymbols.clear();

  void* addr = (void*)m_JIT->getSymbolAddress(GV.getName());

  if (diagnoseUnresolvedSymbols(GV.getName(), "symbol"))
    return 0;
  return addr;
}

bool IncrementalExecutor::diagnoseUnresolvedSymbols(llvm::StringRef trigger,
                                                    llvm::StringRef title) {
  if (m_unresolvedSymbols.empty())
    return false;

  llvm::SmallVector<llvm::Function*, 128> funcsToFree;
  for (std::set<std::string>::const_iterator i = m_unresolvedSymbols.begin(),
         e = m_unresolvedSymbols.end(); i != e; ++i) {
#if 0
    // FIXME: This causes a lot of test failures, for some reason it causes
    // the call to HandleMissingFunction to be elided.
    unsigned diagID = m_Diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                              "%0 unresolved while jitting %1");
    (void)diagID;
    //m_Diags.Report(diagID) << *i << funcname; // TODO: demangle the names.
#endif

    llvm::errs() << "IncrementalExecutor::executeFunction: symbol '" << *i
                 << "' unresolved while linking ";
    if (trigger.find(utils::Synthesize::UniquePrefix) != llvm::StringRef::npos)
      llvm::errs() << "[cling interface function]";
    else {
      if (!title.empty())
        llvm::errs() << title << " '";
      llvm::errs() << trigger;
      if (!title.empty())
        llvm::errs() << "'";
    }
    llvm::errs() << "!\n";

    // Be helpful, demangle!
    std::string demangledName;
    {
#ifndef LLVM_ON_WIN32
      int status = 0;
      char *demang = abi::__cxa_demangle(i->c_str(), 0, 0, &status);
      if (status == 0)
        demangledName = demang;
      free(demang);
#else
      if (char* demang = __unDName(0, i->c_str(), 0, malloc, free, 0)) {
        demangledName = demang;
        free(demang);
      }
#endif
    }
    if (!demangledName.empty()) {
       llvm::errs()
          << "You are probably missing the definition of "
          << demangledName << "\n"
          << "Maybe you need to load the corresponding shared library?\n";
    }

    //llvm::Function *ff = m_engine->FindFunctionNamed(i->c_str());
    // i could also reference a global variable, in which case ff == 0.
    //if (ff)
    //  funcsToFree.push_back(ff);
  }
  //freeCallersOfUnresolvedSymbols(funcsToFree, m_engine.get());
  m_unresolvedSymbols.clear();
  return true;
}

}// end namespace cling
