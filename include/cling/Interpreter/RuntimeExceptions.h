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
    ///\brief Exception that is thrown when a null pointer dereference is found
    /// or a method taking non-null arguments is called with NULL argument.
    /// 
    class cling_null_deref_exception {
    private:
      unsigned m_Location;
      clang::Sema* m_Sema;
    public:
      cling_null_deref_exception(void* Loc, clang::Sema* S);
      ~cling_null_deref_exception();

      void what() throw();

    };    
  } // end namespace runtime
} // end namespace cling
#endif // CLING_RUNTIME_EXCEPTIONS_H 
