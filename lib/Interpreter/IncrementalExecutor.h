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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringRef.h"

#include "IncrementalJIT.h"

#include "cling/Interpreter/Transaction.h"
#include "cling/Interpreter/Value.h"

#include <vector>
#include <set>
#include <map>
#include <memory>
#include <atomic>

namespace clang {
  class DiagnosticsEngine;
}

namespace llvm {
  class GlobalValue;
  class Module;
  class TargetMachine;
}

namespace cling {
  class Value;
  class IncrementalJIT;

  class IncrementalExecutor {
  public:
    typedef void* (*LazyFunctionCreatorFunc_t)(const std::string&);

  private:
    ///\brief Our JIT interface.
    ///
    std::unique_ptr<IncrementalJIT> m_JIT;

    ///\brief Helper that manages when the destructor of an object to be called.
    ///
    /// The object is registered first as an CXAAtExitElement and then cling
    /// takes the control of it's destruction.
    ///
    struct CXAAtExitElement {
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
      CXAAtExitElement(void (*func) (void*), void* arg,
                       const llvm::Module* fromM):
        m_Func(func), m_Arg(arg), m_FromM(fromM) {}

      ///\brief The function to be called.
      ///
      void (*m_Func)(void*);

      ///\brief The single argument passed to the function.
      ///
      void* m_Arg;

      ///\brief The module whose unloading will trigger the call to this atexit
      /// function.
      ///
      const llvm::Module* m_FromM;
    };

    ///\brief Atomic used as a spin lock to protect the access to m_AtExitFuncs
    ///
    /// AddAtExitFunc is used at the end of the 'interpreted' user code
    /// and before the calling framework has any change of taking back/again
    /// its lock protecting the access to cling, so we need to explicit protect
    /// again multiple conccurent access.
    std::atomic_flag m_AtExitFuncsSpinLock; // MSVC doesn't support = ATOMIC_FLAG_INIT;

    typedef llvm::SmallVector<CXAAtExitElement, 128> AtExitFunctions;
    ///\brief Static object, which are bound to unloading of certain declaration
    /// to be destructed.
    ///
    AtExitFunctions m_AtExitFuncs;

    ///\brief Module for which registration of static destructors currently
    /// takes place.
    llvm::Module* m_CurrentAtExitModule;

    ///\brief Modules to emit upon the next call to the JIT.
    ///
    std::vector<llvm::Module*> m_ModulesToJIT;

    ///\brief Lazy function creator, which is a final callback which the
    /// JIT fires if there is unresolved symbol.
    ///
    std::vector<LazyFunctionCreatorFunc_t> m_lazyFuncCreator;

    ///\brief Set of the symbols that the JIT couldn't resolve.
    ///
    std::set<std::string> m_unresolvedSymbols;

#if 0 // See FIXME in IncrementalExecutor.cpp
    ///\brief The diagnostics engine, printing out issues coming from the
    /// incremental executor.
    clang::DiagnosticsEngine& m_Diags;
#endif

    std::unique_ptr<llvm::TargetMachine> CreateHostTargetMachine() const;

  public:
    enum ExecutionResult {
      kExeSuccess,
      kExeFunctionNotCompiled,
      kExeUnresolvedSymbols,
      kNumExeResults
    };

    IncrementalExecutor(clang::DiagnosticsEngine& diags);

    ~IncrementalExecutor();

    void installLazyFunctionCreator(LazyFunctionCreatorFunc_t fp);

    ///\brief Send all collected modules to the JIT, making their symbols
    /// available to jitting (but not necessarily jitting them all).
    Transaction::ExeUnloadHandle emitToJIT() {
      size_t handle = m_JIT->addModules(std::move(m_ModulesToJIT));
      m_ModulesToJIT.clear();
      //m_JIT->finalizeMemory();
      return Transaction::ExeUnloadHandle{(void*)handle};
    }

    ///\brief Unload a set of JIT symbols.
    void unloadFromJIT(llvm::Module* M,
                       Transaction::ExeUnloadHandle H) {
      auto iMod = std::find(m_ModulesToJIT.begin(), m_ModulesToJIT.end(), M);
      if (iMod != m_ModulesToJIT.end())
        m_ModulesToJIT.erase(iMod);
      else
        m_JIT->removeModules((size_t)H.m_Opaque);
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
                                   Value* returnValue = 0) {
      // Set the value to cling::invalid.
      if (returnValue) {
        *returnValue = Value();
      }
      typedef void (*InitFun_t)(void*);
      InitFun_t fun;
      ExecutionResult res = executeInitOrWrapper(function, fun);
      if (res != kExeSuccess)
        return res;
      (*fun)(returnValue);
      return kExeSuccess;
    }

    ///\brief Adds a symbol (function) to the execution engine.
    ///
    /// Allows runtime declaration of a function passing its pointer for being
    /// used by JIT generated code.
    ///
    /// @param[in] symbolName - The name of the symbol as required by the
    ///                         linker (mangled if needed)
    /// @param[in] symbolAddress - The function pointer to register
    /// @returns true if the symbol is successfully registered, false otherwise.
    ///
    bool addSymbol(const char* symbolName,  void* symbolAddress);

    ///\brief Add a llvm::Module to the JIT.
    ///
    /// @param[in] module - The module to pass to the execution engine.
    void addModule(llvm::Module* module) { m_ModulesToJIT.push_back(module); }

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
    void AddAtExitFunc(void (*func) (void*), void* arg);

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
      ExecutionResult res = executeInitOrWrapper(function, fun);
      if (res != kExeSuccess)
        return res;
      (*fun)();
      return kExeSuccess;
    }

    template <class T>
    ExecutionResult executeInitOrWrapper(llvm::StringRef funcname, T& fun) {
      union {
        T fun;
        void* address;
      } p2f;
      p2f.address = (void*)m_JIT->getSymbolAddress(funcname);

      // check if there is any unresolved symbol in the list
      if (diagnoseUnresolvedSymbols(funcname, "function") || !p2f.address) {
        fun = 0;
        return IncrementalExecutor::kExeUnresolvedSymbols;
      }

      fun = p2f.fun;
      return IncrementalExecutor::kExeSuccess;
    }
  };
} // end cling
#endif // CLING_INCREMENTAL_EXECUTOR_H
