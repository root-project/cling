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
  class ASTContext;
  class Decl;
  class QualType;
}

namespace cling {
  namespace runtime {
    namespace internal {
      int local_cxa_atexit(void (*func) (void*), void* arg, void* dso, 
                           void* interp);
    } // end namespace internal
  } // end namespace runtime

  class StoredValueRef;

  class ExecutionContext {
  public:
    typedef void* (*LazyFunctionCreatorFunc_t)(const std::string&);

  private:
    ///\brief Set of the symbols that the ExecutionEngine couldn't resolve.
    /// 
    static std::set<std::string> m_unresolvedSymbols;

    ///\brief Lazy function creator, which is a final callback which the 
    /// ExecutionEngine fires if there is unresolved symbol.
    ///
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


    ///\breif Helper that manages when the destructor of an object to be called.
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
      ///\param [in] dso - The dynamic shared object handle.
      ///\param [in] fromTLD - The unloading of this top level declaration will
      ///                      trigger the atexit function.
      ///
      CXAAtExitElement(void (*func) (void*), void* arg, void* dso,
                       clang::Decl* fromTLD):
        m_Func(func), m_Arg(arg), m_DSO(dso), m_FromTLD(fromTLD) {}

      ///\brief The function to be called.
      ///
      void (*m_Func)(void*);

      ///\brief The single argument passed to the function.
      ///
      void* m_Arg;

      /// \brief The DSO handle.
      ///
      void* m_DSO;

      ///\brief Clang's top level declaration, whose unloading will trigger the
      /// call this atexit function.
      ///
      clang::Decl* m_FromTLD; //FIXME: Should be bound to the llvm symbol.
    };

    ///\brief Static object, which are bound to unloading of certain declaration
    /// to be destructed.
    ///
    std::vector<CXAAtExitElement> m_AtExitFuncs;


  public:
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
    ///\brief We keep track of the entities whose dtor we need to call.
    ///
    int CXAAtExit(void (*func) (void*), void* arg, void* dso, void* clangDecl);

    // This is the caller of CXAAtExit. We want to keep it private so we need
    // to make the caller a friend.
    friend int runtime::internal::local_cxa_atexit(void (*func) (void*), 
                                                   void* arg, void* dso,
                                                   void* interp);
  };
} // end cling
#endif // CLING_EXECUTIONCONTEXT_H
