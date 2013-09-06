//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_RUNTIME_EXCEPTIONS_H
#define CLING_RUNTIME_EXCEPTIONS_H

namespace clang {
  class Sema;
}

namespace cling {
  namespace runtime {
    ///\brief Base class for all interpreter exceptions.
    ///
    class InterpreterException {
    public:
      virtual void what() throw();
    };

    ///\brief Exception that is thrown when a null pointer dereference is found
    /// or a method taking non-null arguments is called with NULL argument.
    /// 
    class NullDerefException : public InterpreterException {
    private:
      unsigned m_Location; // We don't want to #include SourceLocation.h
      clang::Sema* m_Sema;
    public:
      NullDerefException(void* Loc, clang::Sema* S);
      ~NullDerefException();

      virtual void what() throw();

    };    
  } // end namespace runtime
} // end namespace cling
#endif // CLING_RUNTIME_EXCEPTIONS_H 
