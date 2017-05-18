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

#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Utils/Validation.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

extern "C" {
/// Throw an InvalidDerefException if the Arg pointer is invalid.
///\param Interp: The interpreter that has compiled the code.
///\param Expr: The expression corresponding determining the pointer value.
///\param Arg: The pointer to be checked.
///\returns void*, const-cast from Arg, to reduce the complexity in the
/// calling AST nodes, at the expense of possibly doing a
/// T* -> const void* -> const_cast<void*> -> T* round trip.
void* cling_runtime_internal_throwIfInvalidPointer(void* Interp, void* Expr,
                                                   const void* Arg) {

  const clang::Expr* const E = (const clang::Expr*)Expr;

  // The isValidAddress function return true even when the pointer is
  // null thus the checks have to be done before returning successfully from the
  // function in this specific order.
  if (!Arg) {
    cling::Interpreter* I = (cling::Interpreter*)Interp;
    clang::Sema& S = I->getCI()->getSema();
    // Print a nice backtrace.
    I->getCallbacks()->PrintStackTrace();
    throw cling::InvalidDerefException(&S, E,
          cling::InvalidDerefException::DerefType::NULL_DEREF);
  } else if (!cling::utils::isAddressValid(Arg)) {
    cling::Interpreter* I = (cling::Interpreter*)Interp;
    clang::Sema& S = I->getCI()->getSema();
    // Print a nice backtrace.
    I->getCallbacks()->PrintStackTrace();
    throw cling::InvalidDerefException(&S, E,
          cling::InvalidDerefException::DerefType::INVALID_MEM);
  }
  return const_cast<void*>(Arg);
}
}

namespace cling {
  InterpreterException::InterpreterException(const std::string& What) :
    std::runtime_error(What), m_Sema(nullptr) {}
  InterpreterException::InterpreterException(const char* What, clang::Sema* S) :
    std::runtime_error(What), m_Sema(S) {}

  bool InterpreterException::diagnose() const { return false; }
  InterpreterException::~InterpreterException() noexcept {}


  InvalidDerefException::InvalidDerefException(clang::Sema* S,
                                               const clang::Expr* E,
                                               DerefType type)
    : InterpreterException(type == INVALID_MEM  ?
      "Trying to access a pointer that points to an invalid memory address." :
      "Trying to dereference null pointer or trying to call routine taking "
      "non-null arguments", S),
    m_Arg(E), m_Type(type) {}

  InvalidDerefException::~InvalidDerefException() noexcept {}

  bool InvalidDerefException::diagnose() const {
    // Construct custom diagnostic: warning for invalid memory address;
    // no equivalent in clang.
    if (m_Type == cling::InvalidDerefException::DerefType::INVALID_MEM) {
      clang::DiagnosticsEngine& Diags = m_Sema->getDiagnostics();
      unsigned DiagID =
        Diags.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                 "invalid memory pointer passed to a callee:");
      Diags.Report(m_Arg->getLocStart(), DiagID) << m_Arg->getSourceRange();
    }
    else
      m_Sema->Diag(m_Arg->getLocStart(), clang::diag::warn_null_arg)
        << m_Arg->getSourceRange();
    return true;
  }

  CompilationException::CompilationException(const std::string& Reason) :
    InterpreterException(Reason) {}

  CompilationException::~CompilationException() noexcept {}

  void CompilationException::throwingHandler(void * /*user_data*/,
                                             const std::string& reason,
                                             bool /*gen_crash_diag*/) {
    throw cling::CompilationException(reason);
  }
} // end namespace cling
