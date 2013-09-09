//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_COMPILATIONEXCEPTION_H
#define CLING_COMPILATIONEXCEPTION_H

#include <stdexcept>
#include <string>
#include "cling/Interpreter/RuntimeException.h"

namespace cling {
  class Interpreter;
  class MetaProcessor;

  //\brief Exception pull us out of JIT (llvm + clang) errors.
  class CompilationException:
    public virtual runtime::InterpreterException,
    public virtual std::runtime_error {
  public:
    CompilationException(const std::string& reason):
      std::runtime_error(reason) {}
    ~CompilationException() throw(); // vtable pinned to UserInterface.cpp
    virtual const char* what() const throw() {
      return std::runtime_error::what(); }
  };
}

#endif // CLING_COMPILATIONEXCEPTION_H
