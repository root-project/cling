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
    void InterpreterException::what() throw() {
      // For now the default is nop.
    }

    NullDerefException::NullDerefException(void* Loc, clang::Sema* S) 
      : m_Location(*(unsigned *)Loc), m_Sema(S) {}

    NullDerefException::~NullDerefException() {}

    void NullDerefException::what() throw() {
      clang::DiagnosticsEngine& Diag = m_Sema->getDiagnostics();
      Diag.Report(clang::SourceLocation::getFromRawEncoding(m_Location),
                  clang::diag::warn_null_arg);
    }
  } // end namespace runtime
} // end namespace cling
