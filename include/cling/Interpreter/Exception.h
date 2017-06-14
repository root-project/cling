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

#include <stdexcept>

namespace clang {
  class Sema;
  class Expr;
  class DiagnosticsEngine;
}

namespace cling {
  ///\brief Base class for all interpreter exceptions.
  ///
  class InterpreterException : public std::runtime_error {
  protected:
    clang::Sema* const m_Sema;
  public:
    InterpreterException(const std::string& Reason);
    InterpreterException(const char* What, clang::Sema* = nullptr);
    virtual ~InterpreterException() noexcept;

    ///\brief Return true if error was diagnosed false otherwise
    virtual bool diagnose() const;
  };

  ///\brief Exception that is thrown when a invalid pointer dereference is found
  /// or a method taking non-null arguments is called with NULL argument.
  ///
  class InvalidDerefException : public InterpreterException {
  public:
    enum DerefType {INVALID_MEM, NULL_DEREF};
  private:
    const clang::Expr* const m_Arg;
    const DerefType m_Type;
  public:
    InvalidDerefException(clang::Sema* S, const clang::Expr* E, DerefType type);
    virtual ~InvalidDerefException() noexcept;

    bool diagnose() const override;
  };

  ///\brief Exception that pulls cling out of runtime-compilation (llvm + clang)
  ///       errors.
  ///
  /// If user code provokes an llvm::unreachable it will cause this exception
  /// to be thrown.
  /// Note that this exception is *not* thrown during the execution of the
  /// user's code but during its compilation (at runtime).
  class CompilationException: public InterpreterException {
  public:
    CompilationException(const std::string& Reason);
    ~CompilationException() noexcept;

    // Handle fatal llvm errors by throwing an exception.
    // Yes, throwing exceptions in error handlers is bad.
    // Doing nothing is pretty terrible, too.
    static void throwingHandler(void * /*user_data*/,
                                const std::string& reason,
                                bool /*gen_crash_diag*/);
  };
} // end namespace cling

#endif // CLING_RUNTIME_EXCEPTION_H
