//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_EXECUTIONCONTEXT_H
#define CLING_EXECUTIONCONTEXT_H

#include "llvm/ADT/StringRef.h"

#include <vector>
#include <set>

namespace llvm {
  class Module;
  class ExecutionEngine;
  struct GenericValue;
}

namespace clang {
  class CompilerInstance;
  class CodeGenerator;
}

namespace cling {
  class Interpreter;
  class Value;

  class ExecutionContext {
  public:
    typedef void* (*LazyFunctionCreatorFunc_t)(const std::string&);

  public:

    ExecutionContext();
    ~ExecutionContext();

    void installLazyFunctionCreator(LazyFunctionCreatorFunc_t fp);

    void runStaticInitializersOnce(llvm::Module* m);
    void runStaticDestructorsOnce(llvm::Module* m);

    void executeFunction(llvm::StringRef function,
                         llvm::GenericValue* returnValue = 0);

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
      return m_engine;
    }

  private:
    static void* HandleMissingFunction(const std::string&);
    static void* NotifyLazyFunctionCreators(const std::string&);

    int verifyModule(llvm::Module* m);
    void printModule(llvm::Module* m);
    void InitializeBuilder(llvm::Module* m);

    static std::set<std::string> m_unresolvedSymbols;
    static std::vector<LazyFunctionCreatorFunc_t> m_lazyFuncCreator;

    llvm::ExecutionEngine* m_engine; // Owned by JIT

    /// \brief prevent the recursive run of the static inits
    bool m_RunningStaticInits;

    /// \brief Whether cxa_at_exit has been rewired to the Interpreter's version
    bool m_CxaAtExitRemapped;
  };
} // end cling
#endif // CLING_EXECUTIONCONTEXT_H
