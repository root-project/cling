//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------
#ifndef CLING_RUNTIME_UNIVERSE_H
#define CLING_RUNTIME_UNIVERSE_H

#if !defined(__CLING__)
#error "This file must not be included by compiled programs."
#endif

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS // needed by System/DataTypes.h
#endif

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS // needed by System/DataTypes.h
#endif

#ifdef __cplusplus

#ifdef __CLING__CXX11
// FIXME, see http://llvm.org/bugs/show_bug.cgi?id=13530
struct __float128;
#endif

#include "cling/Interpreter/RuntimeException.h"

namespace cling {

  class Interpreter;

  /// \brief Used to stores the declarations, which are going to be
  /// available only at runtime. These are cling runtime builtins
  namespace runtime {

    /// \brief The interpreter provides itself as a builtin, i.e. it
    /// interprets itself. This is particularly important for implementing
    /// the dynamic scopes and the runtime bindings
    extern Interpreter* gCling;

    /// \brief The function is used to deal with null pointer dereference.
    /// It receives input from a user and decides to proceed or not by the
    /// input.
    bool shouldProceed(void* S, void* T);

    namespace internal {
      /// \brief Some of clang's routines rely on valid source locations and
      /// source ranges. This member can be looked up and source locations and
      /// ranges can be passed in as parameters to these routines.
      ///
      /// Use instead of SourceLocation() and SourceRange(). This might help,
      /// when clang emits diagnostics on artificially inserted AST node.
      int InterpreterGeneratedCodeDiagnosticsMaybeIncorrect;

//__cxa_atexit is declared later for WIN32
#if (!_WIN32)
      // Force the module to define __cxa_atexit, we need it.
      struct __trigger__cxa_atexit {
        ~__trigger__cxa_atexit(); // implemented in Interpreter.cpp
      } S;
#endif

    } // end namespace internal
  } // end namespace runtime
} // end namespace cling

using namespace cling::runtime;

// Global d'tors only for C++:
#if _WIN32
extern "C" {

  ///\brief Fake definition to avoid compilation missing function in windows
  /// environment it wont ever be called
  void __dso_handle(){}
  //Fake definition to avoid compilation missing function in windows environment
  //it wont ever be called
  int __cxa_atexit(void (*func) (), void* arg, void* dso) {
    return 0;
  }
}
#endif

extern "C" {
  ///\brief Manually provided by cling missing function resolution using
  /// sys::DynamicLibrary::AddSymbol()
  /// Included in extern C so its name is not mangled and easier to register
  // Implemented in Interpreter.cpp
  int cling__runtime__internal__local_cxa_atexit(void (*func) (void*),
                                                 void* arg,
                                                 void* dso,
                                                 void* interp);
  int cling_cxa_atexit(void (*func) (void*), void* arg, void* dso) {
    return cling__runtime__internal__local_cxa_atexit(func, arg, dso,
                                                 (void*)cling::runtime::gCling);
  }

  ///\Brief a function that throws NullDerefException. This allows to 'hide' the
  /// definition of the exceptions from the RuntimeUniverse and allows us to 
  /// run cling in -no-rtti mode. 
  /// 
  void cling__runtime__internal__throwNullDerefException(void* Sema, 
                                                         void* Expr);

}
#endif // __cplusplus

#endif // CLING_RUNTIME_UNIVERSE_H
