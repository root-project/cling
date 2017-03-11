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

#if !defined(__CLING__) && !defined(CLING_RUNTIME_EXCEPTION_CPP)
#error "This file should only be included at runtime"
#endif

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
  class InterpreterException : public std::runtime_error {
  protected:
    clang::Sema* const m_Sema;
  public:
    InterpreterException(const std::string& Reason);
    InterpreterException(const char* What, clang::Sema* = nullptr);
    virtual ~InterpreterException() LLVM_NOEXCEPT;

    ///\brief Return true if error was diagnosed false otherwise
    virtual bool diagnose() const;

    ///\brief Error handling function type.
    ///
    ///\param[in] Data - User Data passed to RunLoop
    ///\param[in] Err - Pointer to the std::exception or InterpreterException
    ///
    ///\returns Whether the run loop should continue
    ///
    typedef bool (*ErrorHandler)(void* Data, const std::exception* Err);

    ///\brief Default error reporter
    static bool ReportErr(void* Data, const std::exception* Err);

    ///\brief Run Proc(Ptr) until it returns false, catching and reporting
    /// any exceptions that occur.
    ///
    ///\param[in] RunProc - Function to run
    ///\param[in] Ptr - Data to be passed back to RunProc and OnError
    ///\param[in] OnError - Function to run on error
    ///
    static void RunLoop(bool (*RunProc)(void* Data), void* Data,
                        ErrorHandler OnError = &ReportErr);
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
    virtual ~InvalidDerefException() LLVM_NOEXCEPT;

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
    ~CompilationException() LLVM_NOEXCEPT;
  };
} // end namespace cling

#endif // CLING_RUNTIME_EXCEPTION_H
