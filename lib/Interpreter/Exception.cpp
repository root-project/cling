//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Baozeng Ding <sploving1@gmail.com>
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Interpreter/Exception.h"

#include "cling/Utils/Validation.h"

#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/AST/ASTContext.h"

extern "C" {
///\param Arg: Take const void* and return void* to reduce the complexity in the
/// calling AST nodes, at the expense of possibly doing a
/// T* -> const void* -> const_cast<void*> -> T* round trip.
///\param Expr: The Expression to be checked for validity.
///\param Sema: The Sema for the context.
void* cling_runtime_internal_throwIfInvalidPointer(void* Sema, void* Expr,
                                                  const void* Arg) {
  clang::Sema* S = (clang::Sema*)Sema;
  clang::Expr* E = (clang::Expr*)Expr;
  // The isValidAddress function return true even when the pointer is
  // null thus the checks have to be done before returning successfully from the
  // function in this specific order.
  if (!Arg)
    throw cling::InvalidDerefException(S, E,
          cling::InvalidDerefException::DerefType::NULL_DEREF);
  else if (!cling::utils::isAddressValid(Arg))
    throw cling::InvalidDerefException(S, E,
          cling::InvalidDerefException::DerefType::INVALID_MEM);
  return const_cast<void*>(Arg);
}
}

namespace cling {
  // Pin vtable
  InterpreterException::~InterpreterException() {}

  const char* InterpreterException::what() const throw() {
    return "runtime_exception\n";
  }


  InvalidDerefException::InvalidDerefException(clang::Sema* S, clang::Expr* E,
                                  cling::InvalidDerefException::DerefType type)
    : m_Sema(S), m_Arg(E), m_Diags(&m_Sema->getDiagnostics()), m_Type(type) {}

  InvalidDerefException::~InvalidDerefException() {}

  const char* InvalidDerefException::what() const throw() {
    // Invalid memory access.
    if (m_Type == cling::InvalidDerefException::DerefType::INVALID_MEM)
      return "Trying to access a pointer that points to an invalid memory address.";
    // Null deref.
    else
      return "Trying to dereference null pointer or trying to call routine taking non-null arguments";
  }

  void InvalidDerefException::diagnose() const throw() {
    // Construct custom diagnostic: warning for invalid memory address;
    // no equivalent in clang.
    if (m_Type == cling::InvalidDerefException::DerefType::INVALID_MEM) {
      unsigned DiagID =
        m_Diags->getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                 "invalid memory pointer passed to a callee:");
      m_Diags->Report(m_Arg->getLocStart(), DiagID) << m_Arg->getSourceRange();
    }
    else
      m_Sema->Diag(m_Arg->getLocStart(), clang::diag::warn_null_arg)
      << m_Arg->getSourceRange();
  }
} // end namespace cling
