//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Baozeng Ding <sploving1@gmail.com>
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/RuntimeException.h"

#include "cling/Interpreter/Interpreter.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

extern "C" {
void cling__runtime__internal__throwNullDerefException(void* Sema, void* Expr) {
  clang::Sema* S = (clang::Sema*)Sema;
  clang::Expr* E = (clang::Expr*)Expr;

  // FIXME: workaround until JIT supports exceptions
  //throw cling::runtime::NullDerefException(S, E);
  S->Diag(E->getLocStart(), clang::diag::warn_null_arg) << E->getSourceRange();
  if (cling::Interpreter::getNullDerefJump())
    longjmp(*cling::Interpreter::getNullDerefJump(), 1);
}
}

namespace cling {
  namespace runtime {
    // Pin vtable
    InterpreterException::~InterpreterException() {}

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
