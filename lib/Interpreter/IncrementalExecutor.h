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

#include "cling/Interpreter/DynamicLibraryManager.h"
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
  namespace orc {
    class DefinitionGenerator;
  }
}

namespace cling {
  class DynamicLibraryManager;
  class IncrementalJIT;
  class Value;

  class IncrementalExecutor {
  private:
    ///\brief Our JIT interface.
    ///
    std::unique_ptr<IncrementalJIT> m_JIT;

    // optimizer etc passes
    std::unique_ptr<BackendPasses> m_BackendPasses;

    ///\brief Whom to call upon invocation of user code.
    InterpreterCallbacks* m_Callbacks;

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
    /// it's C++ overloads that should be run when a transaction is unloaded.
    ///
    using AtExitFunctions =
      utils::OrderedMap<const Transaction*, std::vector<CXAAtExitElement>>;
    AtExitFunctions m_AtExitFuncs;

    ///\brief Set of the symbols that the JIT couldn't resolve.
    ///
    mutable std::unordered_set<std::string> m_unresolvedSymbols;

#if 0 // See FIXME in IncrementalExecutor.cpp
    ///\brief The diagnostics engine, printing out issues coming from the
    /// incremental executor.
    clang::DiagnosticsEngine& m_Diags;
#endif

    /// Dynamic library manager object.
    ///
    DynamicLibraryManager m_DyLibManager;

  public:
    enum ExecutionResult {
      kExeSuccess,
      kExeFunctionNotCompiled,
      kExeUnresolvedSymbols,
      kNumExeResults
    };

    IncrementalExecutor(clang::DiagnosticsEngine& diags,
                        const clang::CompilerInstance& CI,
                        void *ExtraLibHandle = nullptr,
                        bool Verbose = false);

    ~IncrementalExecutor();

    /// Register a different `IncrementalExecutor` object that can provide
    /// addresses for external symbols.  This is used by child interpreters to
    /// lookup symbols defined in the parent.
    void registerExternalIncrementalExecutor(IncrementalExecutor& IE);

    void setCallbacks(InterpreterCallbacks* callbacks);

    ///\brief Return the LLJIT held by the IncrementalJIT
    llvm::orc::LLJIT* getLLJIT() { return m_JIT ? m_JIT->getLLJIT() : nullptr; }

    const DynamicLibraryManager& getDynamicLibraryManager() const {
      return const_cast<IncrementalExecutor*>(this)->m_DyLibManager;
    }
    DynamicLibraryManager& getDynamicLibraryManager() {
      return m_DyLibManager;
    }

    /// Register a DefinitionGenerator to dynamically provide symbols for
    /// generated code that are not already available within the process.
    void addGenerator(std::unique_ptr<llvm::orc::DefinitionGenerator> G);

    ///\brief Unload a set of JIT symbols.
    llvm::Error unloadModule(const Transaction& T) const {
      return m_JIT->removeModule(T);
    }

    ///\brief Run the static initializers of all modules collected to far.
    ExecutionResult runStaticInitializersOnce(Transaction& T);

    ///\brief Runs all destructors bound to the given transaction and removes
    /// them from the list.
    ///\param[in] T - Transaction to which the dtors were bound.
    ///
    void runAndRemoveStaticDestructors(Transaction* T);

    ///\brief Runs a wrapper function.
    ExecutionResult executeWrapper(llvm::StringRef function,
                                   Value* returnValue = nullptr) const;
    ///\brief Replaces a symbol (function) to the execution engine.
    ///
    /// Allows runtime declaration of a function passing its pointer for being
    /// used by JIT generated code.
    ///
    /// @param[in] Name - The name of the symbol as known by the IR.
    /// @param[in] Address - The function pointer to register
    void replaceSymbol(const char* Name, void* Address) const;

    ///\brief Tells the execution to run all registered atexit functions once.
    ///
    /// This rountine should be used with caution only when an external process
    /// wants to carefully control the teardown. For example, if the process
    /// has registered its own atexit functions which need the interpreter
    /// service to be available when they are being executed.
    ///
    void runAtExitFuncs();

    ///\brief A more meaningful synonym of runAtExitFuncs when used in a more
    /// standard teardown.
    ///
    void shuttingDown() { runAtExitFuncs(); }

    ///\brief Gets the address of an existing global and whether it was JITted.
    ///
    /// JIT symbols might not be immediately convertible to e.g. a function
    /// pointer as their call setup is different.
    ///
    ///\param[in]  mangledName - the global's name
    ///\param[out] fromJIT - whether the symbol was JITted.
    ///
    void*
    getAddressOfGlobal(llvm::StringRef mangledName, bool *fromJIT = nullptr) const;

    ///\brief Return the address of a global from the JIT (as
    /// opposed to dynamic libraries). Forces the emission of the symbol if
    /// it has not happened yet.
    ///
    ///param[in] name - the mangled name of the global.
    void* getPointerToGlobalFromJIT(llvm::StringRef name) const;

    ///\brief Keep track of the entities whose dtor we need to call.
    ///
    void AddAtExitFunc(void (*func)(void*), void* arg, const Transaction* T);

  private:
    ///\brief Emit a llvm::Module to the JIT.
    ///
    /// @param[in] module - The module to pass to the execution engine.
    /// @param[in] optLevel - The optimization level to be used.
    void emitModule(Transaction &T) const {
      if (m_BackendPasses)
        m_BackendPasses->runOnModule(*T.getModule(),
                                     T.getCompilationOpts().OptLevel);

      m_JIT->addModule(T);
    }

    ///\brief Report and empty m_unresolvedSymbols.
    ///\return true if m_unresolvedSymbols was non-empty.
    bool diagnoseUnresolvedSymbols(llvm::StringRef trigger,
                               llvm::StringRef title = llvm::StringRef()) const;

  public:
    ///\brief Remember that the symbol could not be resolved by the JIT.
    void* HandleMissingFunction(const std::string& symbol) const;

  private:
    ///\brief Runs an initializer function.
    ExecutionResult executeInit(llvm::StringRef function) const {
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
    ExecutionResult jitInitOrWrapper(llvm::StringRef funcname, T& fun) const {
      void* fun_ptr = m_JIT->getSymbolAddress(funcname, false /*dlsym*/);

      // check if there is any unresolved symbol in the list
      if (diagnoseUnresolvedSymbols(funcname, "function") || !fun_ptr)
        return IncrementalExecutor::kExeUnresolvedSymbols;

      fun = reinterpret_cast<T>(fun_ptr);
      return IncrementalExecutor::kExeSuccess;
    }
  };
} // end cling
#endif // CLING_INCREMENTAL_EXECUTOR_H
