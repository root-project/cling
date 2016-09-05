//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_RUNTIME_EXCEPTION_H
#define CLING_RUNTIME_EXCEPTION_H

#include "llvm/Support/Compiler.h"
#include <stdexcept>

namespace clang {
  class Sema;
  class Expr;
  class DiagnosticsEngine;
}

namespace cling {
  ///\brief Base class for all interpreter exceptions.
  ///
  class InterpreterException : public std::exception {
  public:
    virtual ~InterpreterException() LLVM_NOEXCEPT;

    virtual const char* what() const LLVM_NOEXCEPT;
    virtual void diagnose() const {}
  };

  ///\brief Exception that is thrown when a invalid pointer dereference is found
  /// or a method taking non-null arguments is called with NULL argument.
  ///
  class InvalidDerefException : public InterpreterException {
  public:
    enum DerefType {INVALID_MEM, NULL_DEREF};
  private:
    clang::Sema* m_Sema;
    clang::Expr* m_Arg;
    clang::DiagnosticsEngine* m_Diags;
    DerefType m_Type;
  public:
    InvalidDerefException(clang::Sema* S, clang::Expr* E, DerefType type);
    virtual ~InvalidDerefException() LLVM_NOEXCEPT;

    const char* what() const LLVM_NOEXCEPT override;
    void diagnose() const override;
  };

  ///\brief Exception that pulls cling out of runtime-compilation (llvm + clang)
  ///       errors.
  ///
  /// If user code provokes an llvm::unreachable it will cause this exception
  /// to be thrown. Given that this is at the process's runtime and an
  /// interpreter error it inherits from InterpreterException and runtime_error.
  /// Note that this exception is *not* thrown during the execution of the
  /// user's code but during its compilation (at runtime).
  class CompilationException: public virtual InterpreterException,
                              public virtual std::runtime_error {
  public:
    CompilationException(const std::string& reason);
    ~CompilationException() LLVM_NOEXCEPT;

    const char* what() const LLVM_NOEXCEPT override;

    // Handle fatal llvm errors by throwing an exception.
    // Yes, throwing exceptions in error handlers is bad.
    // Doing nothing is pretty terrible, too.
    static void throwingHandler(void * /*user_data*/,
                                const std::string& reason,
                                bool /*gen_crash_diag*/);
  };
} // end namespace cling

#endif // CLING_RUNTIME_EXCEPTION_H
