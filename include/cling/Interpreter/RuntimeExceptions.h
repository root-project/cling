//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_RUNTIME_EXCEPTIONS_H
#define CLING_RUNTIME_EXCEPTIONS_H
namespace cling {
  namespace runtime {
    ///\brief Exception that is thrown when a null pointer dereference is found
    /// or a method taking non-null arguments is called with NULL argument.
    /// 
    class cling_null_deref_exception { };    
  } // end namespace runtime
} // end namespace cling
#endif // CLING_RUNTIME_EXCEPTIONS_H 
