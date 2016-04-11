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

#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

namespace cling {
  InvalidDerefException::InvalidDerefException(clang::Sema* S, clang::Expr* E,
                                  cling::InvalidDerefException::DerefType type)
    : m_Sema(S), m_Arg(E), m_Diags(&m_Sema->getDiagnostics()), m_Type(type) {}

  void InvalidDerefException::diagnose() const {
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
