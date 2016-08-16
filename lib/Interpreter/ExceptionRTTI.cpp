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

  clang::Expr* E = (clang::Expr*)Expr;

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
  // Pin vtable
  InterpreterException::~InterpreterException() noexcept {}

  const char* InterpreterException::what() const noexcept {
    return "runtime_exception\n";
  }

  InvalidDerefException::~InvalidDerefException() noexcept {}

  const char* InvalidDerefException::what() const noexcept {
    // Invalid memory access.
    if (m_Type == cling::InvalidDerefException::DerefType::INVALID_MEM)
      return "Trying to access a pointer that points to an invalid memory address.";
    // Null deref.
    else
      return "Trying to dereference null pointer or trying to call routine taking non-null arguments";
  }
} // end namespace cling
