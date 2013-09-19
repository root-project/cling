//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_EXECUTIONCONTEXT_H
#define CLING_EXECUTIONCONTEXT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/OwningPtr.h"

#include <vector>
#include <set>

namespace llvm {
  class Module;
  class ExecutionEngine;
}

namespace clang {
  class QualType;
  class ASTContext;
}

namespace cling {
  class StoredValueRef;

  class ExecutionContext {
  public:
    typedef void* (*LazyFunctionCreatorFunc_t)(const std::string&);

    enum ExecutionResult {
      kExeSuccess,
      kExeFunctionNotCompiled,
      kExeUnresolvedSymbols,
      kNumExeResults
    };

    ExecutionContext(llvm::Module* m);
    ~ExecutionContext();

    void installLazyFunctionCreator(LazyFunctionCreatorFunc_t fp);
    void suppressLazyFunctionCreatorDiags(bool suppressed = true) {
      m_LazyFuncCreatorDiagsSuppressed = suppressed;
    }

    ExecutionResult runStaticInitializersOnce(llvm::Module* m);
    void runStaticDestructorsOnce(llvm::Module* m);

    ExecutionResult executeFunction(llvm::StringRef function,
                                    const clang::ASTContext& Ctx,
                                    clang::QualType retType,
                                    StoredValueRef* returnValue = 0);

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

    ///\brief Gets the address of an existing global and whether it was JITted.
    ///
    /// JIT symbols might not be immediately convertible to e.g. a function
    /// pointer as their call setup is different.
    ///
    ///\param[in]  m       - the module to use for finging the global
    ///\param[in]  mangledName - the globa's name
    ///\param[out] fromJIT - whether the symbol was JITted.
    ///
    void* getAddressOfGlobal(llvm::Module* m, const char* mangledName,
                             bool* fromJIT = 0) const;

    llvm::ExecutionEngine* getExecutionEngine() const {
      return m_engine.get();
    }

  private:
    static void* HandleMissingFunction(const std::string&);
    static void* NotifyLazyFunctionCreators(const std::string&);

    int verifyModule(llvm::Module* m);
    void printModule(llvm::Module* m);
    void InitializeBuilder(llvm::Module* m);

    static std::set<std::string> m_unresolvedSymbols;
    static std::vector<LazyFunctionCreatorFunc_t> m_lazyFuncCreator;

    ///\brief Whether or not the function creator to be queried.
    ///
    static bool m_LazyFuncCreatorDiagsSuppressed;

    ///\brief The llvm ExecutionEngine.
    ///
    llvm::OwningPtr<llvm::ExecutionEngine> m_engine;

    ///\brief prevent the recursive run of the static inits
    ///
    bool m_RunningStaticInits;

    ///\brief Whether cxa_at_exit has been rewired to the Interpreter's version
    ///
    bool m_CxaAtExitRemapped;
  };
} // end cling
#endif // CLING_EXECUTIONCONTEXT_H
