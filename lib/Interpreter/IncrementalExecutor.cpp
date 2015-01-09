//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IncrementalExecutor.h"

#include "cling/Interpreter/Value.h"
#include "cling/Interpreter/Transaction.h"

#include "clang/Basic/Diagnostic.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/PassManager.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DynamicLibrary.h"

using namespace llvm;

namespace {
class ClingMemoryManager: public SectionMemoryManager {
  cling::IncrementalExecutor* m_exe;

  static void local_cxa_atexit(void (*func) (void*), void* arg, void* dso) {
    cling::IncrementalExecutor* exe = (cling::IncrementalExecutor*)dso;
    exe->AddAtExitFunc(func, arg);
  }

public:
  ClingMemoryManager(cling::IncrementalExecutor* Exe):
    m_exe(Exe) {}

  ///\brief Return the address of a symbol, use callbacks if needed.
  uint64_t getSymbolAddress (const std::string &Name) override;
  ///\brief Simply wraps the base class's function setting AbortOnFailure
  /// to false and instead using the error handling mechanism to report it.
  void* getPointerToNamedFunction(const std::string &Name,
                                  bool /*AbortOnFailure*/ =true) override {
    return SectionMemoryManager::getPointerToNamedFunction(Name, false);
  }
};

uint64_t ClingMemoryManager::getSymbolAddress(const std::string &Name) {
  if (Name == "__cxa_atexit") {
    // Rewrire __cxa_atexit to ~Interpreter(), thus also global destruction
    // coming from the JIT.
    return (uint64_t)&local_cxa_atexit;
  } else if (Name == "__dso_handle") {
    // Provide IncrementalExecutor as the third argument to __cxa_atexit.
    return (uint64_t)m_exe;
  }
  if (uint64_t Addr = SectionMemoryManager::getSymbolAddress(Name))
    return Addr;
  return (uint64_t) m_exe->NotifyLazyFunctionCreators(Name);
}

}

namespace cling {

std::set<std::string> IncrementalExecutor::m_unresolvedSymbols;

std::vector<IncrementalExecutor::LazyFunctionCreatorFunc_t>
  IncrementalExecutor::m_lazyFuncCreator;

// Keep in source: OwningPtr<ExecutionEngine> needs #include ExecutionEngine
  IncrementalExecutor::IncrementalExecutor(llvm::Module* m,
                                           clang::DiagnosticsEngine& /*diags*/):
    m_CurrentAtExitModule(0)
#if 0
    : m_Diags(diags)
#endif
{
  assert(m && "llvm::Module must not be null!");
  m_AtExitFuncs.reserve(256);

  //
  //  Create an execution engine to use.
  //
  assert(m && "Module cannot be null");

  // Note: Engine takes ownership of the module.
  llvm::EngineBuilder builder(m);

  std::string errMsg;
  builder.setErrorStr(&errMsg);
  builder.setOptLevel(llvm::CodeGenOpt::Less);
  builder.setEngineKind(llvm::EngineKind::JIT);
  builder.setUseMCJIT(true);
  builder.setMCJITMemoryManager(new ClingMemoryManager(this));

  // EngineBuilder uses default c'ted TargetOptions, too:
  llvm::TargetOptions TargetOpts;
  TargetOpts.NoFramePointerElim = 1;
  TargetOpts.JITEmitDebugInfo = 1;

  builder.setTargetOptions(TargetOpts);

  m_engine.reset(builder.create());
  assert(m_engine && "Cannot create module!");

  // install lazy function creators
  //m_engine->InstallLazyFunctionCreator(NotifyLazyFunctionCreators);
}

// Keep in source: ~OwningPtr<ExecutionEngine> needs #include ExecutionEngine
IncrementalExecutor::~IncrementalExecutor() {}

void IncrementalExecutor::shuttingDown() {
  for (size_t I = 0, N = m_AtExitFuncs.size(); I < N; ++I) {
    const CXAAtExitElement& AEE = m_AtExitFuncs[N - I - 1];
    (*AEE.m_Func)(AEE.m_Arg);
  }
}

void IncrementalExecutor::remapSymbols() {
  // Note: iteration of ++remapI happens in the body due to invalidation
  // of the erased iterator!
  for (auto remapI = std::begin(m_SymbolsToRemap),
         remapE = std::end(m_SymbolsToRemap);
       remapI != remapE;) {
    // The function for which the symbol address will be replaced
    llvm::Function* origFunc
      = m_engine->FindFunctionNamed(remapI->first.c_str());
    if (!origFunc) {
      // Go to next element.
      ++remapI;
      continue;
    }

    // The new symbol address, which might be NULL to signal a symbol
    // lookup is required
    void* replaceAddr = remapI->second.first;
    if (!replaceAddr) {
      // A symbol lookup is required to find the replacement address.
      llvm::Function* interpFunc
        = m_engine->FindFunctionNamed(remapI->second.second.c_str());
      assert(interpFunc && "replacement function must exist.");
      // Generate the symbol and get its address
      replaceAddr = m_engine->getPointerToFunction(interpFunc);
    }
    assert(replaceAddr && "cannot find replacement symbol");
    // Replace the mapping of function symbol to new address
    m_engine->updateGlobalMapping(origFunc, replaceAddr);

    // Note that the current entry was successfully remapped.
    // Save the current so we can erase it *after* the iterator increment
    // or we would increment an invalid iterator.
    auto remapErase = remapI;
    ++remapI;
    m_SymbolsToRemap.erase(remapErase);
  }
}

void IncrementalExecutor::AddAtExitFunc(void (*func) (void*), void* arg) {
  // Register a CXAAtExit function
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

static void
freeCallersOfUnresolvedSymbols(llvm::SmallVectorImpl<llvm::Function*>&
                               funcsToFree, llvm::ExecutionEngine* engine) {
  llvm::SmallPtrSet<llvm::Function*, 40> funcsToFreeUnique;
  for (size_t i = 0; i < funcsToFree.size(); ++i) {
    llvm::Function* func = funcsToFree[i];
    assert(func && "Cannot free NULL function");
    if (funcsToFreeUnique.insert(func)) {
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

IncrementalExecutor::ExecutionResult
IncrementalExecutor::executeFunction(llvm::StringRef funcname,
                                     Value* returnValue) {
  // Call a function without arguments, or with an SRet argument, see SRet below
  // We don't care whether something was unresolved before.
  m_unresolvedSymbols.clear();

  // Set the value to cling::invalid.
  if (returnValue) {
    *returnValue = Value();
  }
  remapSymbols();

  llvm::Function* f = m_engine->FindFunctionNamed(funcname.str().c_str());
  if (!f) {
    llvm::errs() << "IncrementalExecutor::executeFunction: "
      "could not find function named " << funcname << '\n';
    return kExeFunctionNotCompiled;
  }
  assert (f->getFunctionType()->getNumParams() == 1
          && (*f->getFunctionType()->param_begin())->isPtrOrPtrVectorTy() &&
          "Wrong signature");
  typedef void (*PromptWrapper_t)(void*);
  union {
    PromptWrapper_t wrapperFunction;
    void* address;
  } p2f;
  p2f.address = (void*)m_engine->getFunctionAddress(funcname);

  // check if there is any unresolved symbol in the list
  if (diagnoseUnresolvedSymbols(funcname, "function"))
    return kExeUnresolvedSymbols;

  // Run the function
  (*p2f.wrapperFunction)(returnValue);

  return kExeSuccess;
}

IncrementalExecutor::ExecutionResult
IncrementalExecutor::runStaticInitializersOnce(llvm::Module* m) {
  assert(m && "Module must not be null");
  assert(m_engine && "Code generation did not create an engine!");

  // Set m_CurrentAtExitModule to the Module, unset to 0 once done.
  struct AtExitModuleSetterRAII {
    llvm::Module*& m_AEM;
    AtExitModuleSetterRAII(llvm::Module* M, llvm::Module*& AEM): m_AEM(AEM)
    { AEM = M; }
    ~AtExitModuleSetterRAII() { m_AEM = 0; }
  } DSOHandleSetter(m, m_CurrentAtExitModule);

  m_engine->finalizeObject();

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

  // We don't care whether something was unresolved before.
  m_unresolvedSymbols.clear();

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
      remapSymbols();
      m_engine->getPointerToFunction(F);
      // check if there is any unresolved symbol in the list
      if (diagnoseUnresolvedSymbols("static initializers"))
        return kExeUnresolvedSymbols;

      //executeFunction(F->getName());
      m_engine->runFunction(F, std::vector<llvm::GenericValue>());
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
  for (AtExitFunctions::iterator I = m_AtExitFuncs.begin();
       I != m_AtExitFuncs.end();)
    if (I->m_FromM == T->getModule()) {
      boundToT.push_back(*I);
      I = m_AtExitFuncs.erase(I);
    }
    else
      ++I;

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

void IncrementalExecutor::addModule(llvm::Module* module) {
  m_engine->addModule(module);
}


void* IncrementalExecutor::getAddressOfGlobal(llvm::Module* m,
                                              llvm::StringRef symbolName,
                                              bool* fromJIT /*=0*/) {
  // Return a symbol's address, and whether it was jitted.
  void* address
    = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(symbolName);
  if (address) {
    if (fromJIT) *fromJIT = false;
  } else {
    if (fromJIT) *fromJIT = true;
    llvm::GlobalVariable* gvar = m->getGlobalVariable(symbolName, true);
    if (!gvar)
      return 0;

    remapSymbols();
    address = m_engine->getPointerToGlobal(gvar);
  }
  return address;
}

void*
IncrementalExecutor::getPointerToGlobalFromJIT(const llvm::GlobalValue& GV) {
  // Get the function / variable pointer referenced by GV.

  // We don't care whether something was unresolved before.
  m_unresolvedSymbols.clear();

  remapSymbols();
  if (void* addr = m_engine->getPointerToGlobalIfAvailable(&GV))
    return addr;

  //  Function not yet codegened by the JIT, force this to happen now.
  void* Ptr = m_engine->getPointerToGlobal(&GV);
  if (diagnoseUnresolvedSymbols(GV.getName(), "symbol"))
    return 0;
  return Ptr;
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
    if (!title.empty())
      llvm::errs() << title << "'";
    llvm::errs() << trigger;
    if (!title.empty())
      llvm::errs() << "'";
    llvm::errs() << "!\n";
    llvm::Function *ff = m_engine->FindFunctionNamed(i->c_str());
    // i could also reference a global variable, in which case ff == 0.
    if (ff)
      funcsToFree.push_back(ff);
  }
  freeCallersOfUnresolvedSymbols(funcsToFree, m_engine.get());
  m_unresolvedSymbols.clear();
  return true;
}

}// end namespace cling
