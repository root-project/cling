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
    cling_null_deref_exception::cling_null_deref_exception(
      void* Loc, clang::Sema* S) : m_Location(*(unsigned *)Loc), m_Sema(S){}

    cling_null_deref_exception::~cling_null_deref_exception() {}

    void cling_null_deref_exception::what() throw() {
        clang::DiagnosticsEngine& Diag = m_Sema->getDiagnostics();
        Diag.Report(clang::SourceLocation::getFromRawEncoding(m_Location),
                                              clang::diag::warn_null_arg);
    }
  } // end namespace runtime
} // end namespace cling
