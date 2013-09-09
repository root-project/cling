//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Baozeng Ding <sploving1@gmail.com>
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/RuntimeExceptions.h"

#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

namespace cling {
  namespace runtime {
    const char* InterpreterException::what() const throw() {
      return "runtime_exception\n";
    }

    NullDerefException::NullDerefException(clang::Sema* S, clang::Expr* E)
      : m_Sema(S), m_Arg(E) {}

    NullDerefException::~NullDerefException() {}

    const char* NullDerefException::what() const throw() {
      return "Trying to dereference null pointer or trying to call routine taking non-null arguments";
    }
    
    void NullDerefException::diagnose() const throw() {
      m_Sema->Diag(m_Arg->getLocStart(), clang::diag::warn_null_arg)
        << m_Arg->getSourceRange();
    }
  } // end namespace runtime
} // end namespace cling
