//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_INCREMENTAL_EXECUTOR_H
#define CLING_INCREMENTAL_EXECUTOR_H

#include "IncrementalJIT.h"

#include "BackendPasses.h"
#include "EnterUserCodeRAII.h"

#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Interpreter/Value.h"
#include "cling/Utils/Casting.h"
#include "cling/Utils/OrderedMap.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringRef.h"

#include <atomic>
#include <map>
#include <memory>
#include <unordered_set>
#include <vector>

namespace clang {
  class DiagnosticsEngine;
  class CodeGenOptions;
  class CompilerInstance;
}

namespace llvm {
  class GlobalValue;
  class Module;
  class TargetMachine;
}

namespace cling {
  class IncrementalJIT;
  class Value;

  class IncrementalExecutor {
  public:
    typedef void* (*LazyFunctionCreatorFunc_t)(const std::string&);

  private:
    ///\brief Our JIT interface.
    ///
    std::unique_ptr<IncrementalJIT> m_JIT;

    // optimizer etc passes
    std::unique_ptr<BackendPasses> m_BackendPasses;

    ///\brief Whom to call upon invocation of user code.
    InterpreterCallbacks* m_Callbacks;

    ///\brief A pointer to the IncrementalExecutor of the parent Interpreter.
    ///
    IncrementalExecutor* m_externalIncrementalExecutor;

    ///\brief Helper that manages when the destructor of an object to be called.
    ///
    /// The object is registered first as an CXAAtExitElement and then cling
    /// takes the control of it's destruction.
    ///
    class CXAAtExitElement {
      ///\brief The function to be called.
      ///
      void (*m_Func)(void*);

      ///\brief The single argument passed to the function.
      ///
      void* m_Arg;

    public:
      ///\brief Constructs an element, whose destruction time will be managed by
      /// the interpreter. (By registering a function to be called by exit
      /// or when a shared library is unloaded.)
      ///
      /// Registers destructors for objects with static storage duration with
      /// the _cxa atexit function rather than the atexit function. This option
      /// is required for fully standards-compliant handling of static
      /// destructors(many of them created by cling), but will only work if
      /// your C library supports __cxa_atexit (means we have our own work
      /// around for Windows). More information about __cxa_atexit could be
      /// found in the Itanium C++ ABI spec.
      ///
      ///\param [in] func - The function to be called on exit or unloading of
      ///                   shared lib.(The destructor of the object.)
      ///\param [in] arg - The argument the func to be called with.
      ///\param [in] fromT - The unloading of this transaction will trigger the
      ///                    atexit function.
      ///
      CXAAtExitElement(void (*func)(void*), void* arg)
          : m_Func(func), m_Arg(arg) {}

      void operator()() const { (*m_Func)(m_Arg); }
    };

    ///\brief Atomic used as a spin lock to protect the access to m_AtExitFuncs
    ///
    /// AddAtExitFunc is used at the end of the 'interpreted' user code
    /// and before the calling framework has any change of taking back/again
    /// its lock protecting the access to cling, so we need to explicit protect
    /// again multiple conccurent access.
    std::atomic_flag m_AtExitFuncsSpinLock; // MSVC doesn't support = ATOMIC_FLAG_INIT;

    ///\brief Function registered via __cxa_atexit, atexit, or one of
    /// it's C++ overloads that should be run when a module is unloaded.
    ///
    // FIXME: We should probably try using a weak_ptr instead of a shared_ptr.
    typedef utils::OrderedMap<std::shared_ptr<llvm::Module>,
                              std::vector<CXAAtExitElement>>
        AtExitFunctions;
    AtExitFunctions m_AtExitFuncs;

    ///\brief Modules to emit upon the next call to the JIT.
    ///
    std::vector<llvm::Module*> m_ModulesToJIT;

    ///\brief Lazy function creator, which is a final callback which the
    /// JIT fires if there is unresolved symbol.
    ///
    std::vector<LazyFunctionCreatorFunc_t> m_lazyFuncCreator;

    ///\brief Set of the symbols that the JIT couldn't resolve.
    ///
    std::unordered_set<std::string> m_unresolvedSymbols;

#if 0 // See FIXME in IncrementalExecutor.cpp
    ///\brief The diagnostics engine, printing out issues coming from the
    /// incremental executor.
    clang::DiagnosticsEngine& m_Diags;
#endif


  public:
    enum ExecutionResult {
      kExeSuccess,
      kExeFunctionNotCompiled,
      kExeUnresolvedSymbols,
      kNumExeResults
    };

    IncrementalExecutor(clang::DiagnosticsEngine& diags,
                        const clang::CompilerInstance& CI);

    ~IncrementalExecutor();

    void setExternalIncrementalExecutor(IncrementalExecutor *extIncrExec) {
      m_externalIncrementalExecutor = extIncrExec;
    }
    void setCallbacks(InterpreterCallbacks* callbacks) {
      m_Callbacks = callbacks;
    }
    void installLazyFunctionCreator(LazyFunctionCreatorFunc_t fp);

    ///\brief Unload a set of JIT symbols.
    bool unloadModule(const std::shared_ptr<llvm::Module>& M) {
      // FIXME: Propagate the error in a more verbose way.
      if (auto Err = m_JIT->removeModule(M))
        return false;
      return true;
    }

    ///\brief Run the static initializers of all modules collected to far.
    ExecutionResult runStaticInitializersOnce(const Transaction& T);

    ///\brief Runs all destructors bound to the given transaction and removes
    /// them from the list.
    ///\param[in] T - Transaction to which the dtors were bound.
    ///
    void runAndRemoveStaticDestructors(Transaction* T);

    ///\brief Runs a wrapper function.
    ExecutionResult executeWrapper(llvm::StringRef function,
                                   Value* returnValue = 0);
    ///\brief Adds a symbol (function) to the execution engine.
    ///
    /// Allows runtime declaration of a function passing its pointer for being
    /// used by JIT generated code.
    ///
    /// @param[in] Name - The name of the symbol as required by the
    ///                         linker (mangled if needed)
    /// @param[in] Address - The function pointer to register
    /// @param[in] JIT - Add to the JIT injected symbol table
    /// @returns true if the symbol is successfully registered, false otherwise.
    ///
    bool addSymbol(const char* Name, void* Address, bool JIT = false);

    ///\brief Emit a llvm::Module to the JIT.
    ///
    /// @param[in] module - The module to pass to the execution engine.
    /// @param[in] optLevel - The optimization level to be used.
    void emitModule(const std::shared_ptr<llvm::Module>& module, int optLevel) {
      if (m_BackendPasses)
        m_BackendPasses->runOnModule(*module, optLevel);

      m_JIT->addModule(module);
    }

    ///\brief Tells the execution context that we are shutting down the system.
    ///
    /// This that notification is needed because the execution context needs to
    /// perform extra actions like delete all managed by it symbols, which might
    /// still require alive system.
    ///
    void shuttingDown();

    ///\brief Gets the address of an existing global and whether it was JITted.
    ///
    /// JIT symbols might not be immediately convertible to e.g. a function
    /// pointer as their call setup is different.
    ///
    ///\param[in]  mangledName - the globa's name
    ///\param[out] fromJIT - whether the symbol was JITted.
    ///
    void* getAddressOfGlobal(llvm::StringRef mangledName, bool* fromJIT = 0);

    ///\brief Return the address of a global from the JIT (as
    /// opposed to dynamic libraries). Forces the emission of the symbol if
    /// it has not happened yet.
    ///
    ///param[in] GV - global value for which the address will be returned.
    void* getPointerToGlobalFromJIT(const llvm::GlobalValue& GV);

    ///\brief Keep track of the entities whose dtor we need to call.
    ///
    void AddAtExitFunc(void (*func)(void*), void* arg,
                       const std::shared_ptr<llvm::Module>& M);

    ///\brief Try to resolve a symbol through our LazyFunctionCreators;
    /// print an error message if that fails.
    void* NotifyLazyFunctionCreators(const std::string&);

  private:
    ///\brief Report and empty m_unresolvedSymbols.
    ///\return true if m_unresolvedSymbols was non-empty.
    bool diagnoseUnresolvedSymbols(llvm::StringRef trigger,
                                   llvm::StringRef title = llvm::StringRef());

    ///\brief Remember that the symbol could not be resolved by the JIT.
    void* HandleMissingFunction(const std::string& symbol);

    ///\brief Runs an initializer function.
    ExecutionResult executeInit(llvm::StringRef function) {
      typedef void (*InitFun_t)();
      InitFun_t fun;
      ExecutionResult res = jitInitOrWrapper(function, fun);
      if (res != kExeSuccess)
        return res;
      EnterUserCodeRAII euc(m_Callbacks);
      (*fun)();
      return kExeSuccess;
    }

    template <class T>
    ExecutionResult jitInitOrWrapper(llvm::StringRef funcname, T& fun) {
      fun = utils::UIntToFunctionPtr<T>(m_JIT->getSymbolAddress(funcname,
                                                              false /*dlsym*/));

      // check if there is any unresolved symbol in the list
      if (diagnoseUnresolvedSymbols(funcname, "function") || !fun)
        return IncrementalExecutor::kExeUnresolvedSymbols;

      return IncrementalExecutor::kExeSuccess;
    }
  };
} // end cling
#endif // CLING_INCREMENTAL_EXECUTOR_H
