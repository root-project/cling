//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------
#ifndef __CLING__
#error "This file must not be included by compiled programs."
#else

#ifdef CLING_RUNTIME_UNIVERSE_H
#error "CLING_RUNTIME_UNIVERSE_H Must only include once."
#else

#define CLING_RUNTIME_UNIVERSE_H

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS // needed by System/DataTypes.h
#endif

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS // needed by System/DataTypes.h
#endif

#ifdef __cplusplus

namespace cling {

  class Interpreter;

  /// \brief Used to stores the declarations, which are going to be
  /// available only at runtime. These are cling runtime builtins
  namespace runtime {

    /// \brief The interpreter provides itself as a builtin, i.e. it
    /// interprets itself. This is particularly important for implementing
    /// the dynamic scopes and the runtime bindings
    Interpreter* gCling = 0;

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
}
#endif // __cplusplus

#endif // CLING_RUNTIME_UNIVERSE_H (error)

#endif // __CLING__
