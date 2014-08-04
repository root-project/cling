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
}

namespace cling {
  namespace runtime {
    ///\brief Base class for all interpreter exceptions.
    ///
    class InterpreterException {
    public:
      virtual const char* what() const throw();
      virtual ~InterpreterException();
    };

    ///\brief Exception that is thrown when a null pointer dereference is found
    /// or a method taking non-null arguments is called with NULL argument.
    ///
    class NullDerefException : public InterpreterException {
    private:
      clang::Sema* m_Sema;
      clang::Expr* m_Arg;
    public:
      NullDerefException(clang::Sema* S, clang::Expr* E);
      virtual ~NullDerefException();

      virtual const char* what() const throw();
      void diagnose() const throw();
    };
  } // end namespace runtime
} // end namespace cling
#endif // CLING_RUNTIME_EXCEPTION_H
