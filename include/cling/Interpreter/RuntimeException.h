//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
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
      ~NullDerefException();

      virtual const char* what() const throw();
      void diagnose() const throw();
    };
  } // end namespace runtime
} // end namespace cling
#endif // CLING_RUNTIME_EXCEPTION_H 
