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

namespace clang {
  class Sema;
  class Expr;
  class DiagnosticsEngine;
}

namespace cling {
  ///\brief Base class for all interpreter exceptions.
  ///
  class InterpreterException {
  public:
    virtual const char* what() const throw();
    virtual ~InterpreterException();
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
    virtual ~InvalidDerefException();

    virtual const char* what() const throw();
    void diagnose() const throw();
  };
} // end namespace cling
#endif // CLING_RUNTIME_EXCEPTION_H
