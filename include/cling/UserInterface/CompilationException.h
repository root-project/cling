//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_COMPILATIONEXCEPTION_H
#define CLING_COMPILATIONEXCEPTION_H

#include <stdexcept>
#include <string>
#include "cling/Interpreter/RuntimeException.h"

namespace cling {
  class Interpreter;
  class MetaProcessor;

  ///\brief Exception that pulls cling out of runtime-compilation (llvm + clang)
  ///       errors.
  ///
  /// If user code provokes an llvm::unreachable it will cause this exception
  /// to be thrown. Given that this is at the process's runtime and an
  /// interpreter error it inherits from InterpreterException and runtime_error.
  /// Note that this exception is *not* thrown during the execution of the
  /// user's code but during its compilation (at runtime).
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
