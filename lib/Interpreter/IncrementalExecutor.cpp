//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "IncrementalExecutor.h"

#include "clang/AST/Type.h"
#include "clang/Sema/Sema.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/PassManager.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DynamicLibrary.h"

using namespace cling;

std::set<std::string> IncrementalExecutor::m_unresolvedSymbols;
std::vector<IncrementalExecutor::LazyFunctionCreatorFunc_t>
  IncrementalExecutor::m_lazyFuncCreator;

// Keep in source: OwningPtr<ExecutionEngine> needs #include ExecutionEngine
IncrementalExecutor::IncrementalExecutor(llvm::Module* m) 
  : m_CxaAtExitRemapped(false)
{
  assert(m && "llvm::Module must not be null!");
  m_AtExitFuncs.reserve(256);
  InitializeBuilder(m);
}

// Keep in source: ~OwningPtr<ExecutionEngine> needs #include ExecutionEngine
IncrementalExecutor::~IncrementalExecutor() {}

void IncrementalExecutor::shuttingDown() {
  for (size_t I = 0, N = m_AtExitFuncs.size(); I < N; ++I) {
    const CXAAtExitElement& AEE = m_AtExitFuncs[N - I - 1];
    (*AEE.m_Func)(AEE.m_Arg);
  }
}

void IncrementalExecutor::remapCXAAtExit() {
  assert(!m_CxaAtExitRemapped && "__cxa_at_exit already remapped.");
  llvm::Function* clingAtExit
    = m_engine->FindFunctionNamed("cling_cxa_atexit");
  assert(clingAtExit && "cling_cxa_atexit must exist.");

  llvm::Function* atExit = m_engine->FindFunctionNamed("__cxa_atexit");
  if (!atExit) {
    // Inject __cxa_atexit into module
    llvm::Type* retTy = 0;
    llvm::Type* voidPtrTy = 0;
    if (sizeof(int) == 4) {
      retTy = llvm::Type::getInt32Ty(llvm::getGlobalContext());
      voidPtrTy = llvm::Type::getInt32PtrTy(llvm::getGlobalContext());
    } else if (sizeof(int) == 8) {
      retTy = llvm::Type::getInt64Ty(llvm::getGlobalContext());
      voidPtrTy = llvm::Type::getInt64PtrTy(llvm::getGlobalContext());
    } else {
      assert(retTy && "Unsupported sizeof(int)!");
      retTy = llvm::Type::getInt64Ty(llvm::getGlobalContext());
      voidPtrTy = llvm::Type::getInt64PtrTy(llvm::getGlobalContext());
    }

    llvm::SmallVector<llvm::Type*, 3> argTy;
    argTy.push_back(voidPtrTy);
    argTy.push_back(voidPtrTy);
    argTy.push_back(voidPtrTy);
    llvm::FunctionType* cxaatexitTy
      = llvm::FunctionType::get(retTy, argTy, false /*varArg*/);
    llvm::Function* atexitFunc
      = llvm::Function::Create(cxaatexitTy, llvm::GlobalValue::InternalLinkage,
                               "__cxa_atexit", 0 /*module*/);
    m_engine->addGlobalMapping(atexitFunc, clingAtExit);
  }

  void* clingAtExitAddr = m_engine->getPointerToFunction(clingAtExit);
  assert(clingAtExitAddr && "cannot find cling_cxa_atexit");
  m_engine->updateGlobalMapping(atExit, clingAtExitAddr);
  m_CxaAtExitRemapped = true;
}

void IncrementalExecutor::InitializeBuilder(llvm::Module* m) {
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
  builder.setAllocateGVsWithCode(false);

  // EngineBuilder uses default c'ted TargetOptions, too:
  llvm::TargetOptions TargetOpts;
  TargetOpts.NoFramePointerElim = 1;
  TargetOpts.JITEmitDebugInfo = 1;

  builder.setTargetOptions(TargetOpts);

  m_engine.reset(builder.create());
  if (!m_engine)
     llvm::errs() << "cling::IncrementalExecutor::InitializeBuilder(): "
                  << errMsg;
  assert(m_engine && "Cannot create module!");

  // install lazy function creators
  m_engine->InstallLazyFunctionCreator(NotifyLazyFunctionCreators);
}

int IncrementalExecutor::CXAAtExit(void (*func) (void*), void* arg, void* dso,
                                void* clingT) {
  // Register a CXAAtExit function
  cling::Transaction* T = (cling::Transaction*)clingT;
  m_AtExitFuncs.push_back(CXAAtExitElement(func, arg, dso, T));
  return 0; // happiness
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
    llvm::errs() << "IncrementalExecutor: use of undefined symbol '"
                 << mangled_name << "'!\n";
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
                                  StoredValueRef* returnValue)
{
  // Call a function without arguments, or with an SRet argument, see SRet below
  // We don't care whether something was unresolved before.
  m_unresolvedSymbols.clear();

  llvm::Function* f = m_engine->FindFunctionNamed(funcname.str().c_str());
  if (!f) {
    llvm::errs() << "IncrementalExecutor::executeFunction: "
      "could not find function named " << funcname << '\n';
    return kExeFunctionNotCompiled;
  }
  typedef void (*PromptWrapper_t)(void*);
  union {
    PromptWrapper_t wrapperFunction;
    void* address;
  } p2f;
  p2f.address = m_engine->getPointerToFunction(f);

  // check if there is any unresolved symbol in the list
  if (!m_unresolvedSymbols.empty()) {
    llvm::SmallVector<llvm::Function*, 100> funcsToFree;
    for (std::set<std::string>::const_iterator i = m_unresolvedSymbols.begin(),
           e = m_unresolvedSymbols.end(); i != e; ++i) {
      llvm::errs() << "IncrementalExecutor::executeFunction: symbol '" << *i
                   << "' unresolved while linking function '" << funcname
                   << "'!\n";
      llvm::Function *ff = m_engine->FindFunctionNamed(i->c_str());
      // i could also reference a global variable, in which case ff == 0.
      if (ff)
        funcsToFree.push_back(ff);
    }
    freeCallersOfUnresolvedSymbols(funcsToFree, m_engine.get());
    m_unresolvedSymbols.clear();
    return kExeUnresolvedSymbols;
  }

  // Run the function
  (*p2f.wrapperFunction)(returnValue);

  return kExeSuccess;
}


IncrementalExecutor::ExecutionResult
IncrementalExecutor::runStaticInitializersOnce(llvm::Module* m) {
  assert(m && "Module must not be null");
  assert(m_engine && "Code generation did not create an engine!");

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

  GV->eraseFromParent();

  if (InitList == 0)
    return kExeSuccess;

  // We don't care whether something was unresolved before.
  m_unresolvedSymbols.clear();

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
      m_engine->getPointerToFunction(F);
      // check if there is any unresolved symbol in the list
      if (!m_unresolvedSymbols.empty()) {
        llvm::SmallVector<llvm::Function*, 100> funcsToFree;
        for (std::set<std::string>::const_iterator i = m_unresolvedSymbols.begin(),
               e = m_unresolvedSymbols.end(); i != e; ++i) {
          llvm::errs() << "IncrementalExecutor::runStaticInitializersOnce: symbol '" << *i
                       << "' unresolved while linking static initializer '"
                       << F->getName() << "'!\n";
          llvm::Function *ff = m_engine->FindFunctionNamed(i->c_str());
          assert(ff && "cannot find function to free");
          funcsToFree.push_back(ff);
        }
        freeCallersOfUnresolvedSymbols(funcsToFree, m_engine.get());
        m_unresolvedSymbols.clear();
        return kExeUnresolvedSymbols;
      }
      m_engine->runFunction(F, std::vector<llvm::GenericValue>());
    }
  }

  return kExeSuccess;
}

void IncrementalExecutor::runAndRemoveStaticDestructors(Transaction* T) {
  assert(T && "Must be set");
  // Collect all the dtors bound to this transaction.
  AtExitFunctions boundToT;
  for (AtExitFunctions::iterator I = m_AtExitFuncs.begin();
       I != m_AtExitFuncs.end();)
    if (I->m_FromT == T) {
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

void* IncrementalExecutor::getAddressOfGlobal(llvm::Module* m,
                                              const char* symbolName,
                                              bool* fromJIT /*=0*/) const {
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

    address = m_engine->getPointerToGlobal(gvar);
  }
  return address;
}

void*
IncrementalExecutor::getPointerToGlobalFromJIT(const llvm::GlobalValue& GV)const{
  if (void* addr = m_engine->getPointerToGlobalIfAvailable(&GV))
    return addr;

  //  Function not yet codegened by the JIT, force this to happen now.
  return m_engine->getPointerToGlobal(&GV);
}
