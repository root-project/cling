//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author: Roman Zulak
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Diagnostics.h"

namespace cling {
namespace utils {

ReplaceDiagnostics::ReplaceDiagnostics(clang::DiagnosticsEngine& D,
                         clang::DiagnosticConsumer& Replace, bool Own) :
  m_Diags(D), m_PrevClient(*D.getClient()), m_PrevOwn(D.ownsClient()) {
  // DiagnosticsEngine requires a client to be set, so guarantee
  // m_PrevClient is not null by it being a reference.
  assert(D.getClient() && "DiagnosticConsumer not set");
  // Take the std::unique_ptr and release it, we have the raw one
  D.takeClient().release();
  // Set the new client / consumer
  D.setClient(&Replace, Own);
}

namespace {
  enum {
      kReport = 1,  ///< Report errors on destruction
      kReset  = 2,  ///< Reset DiagnosticsEngine on destruction
  };
}

DiagnosticsStore::DiagnosticsStore(clang::DiagnosticsEngine& Diags, bool Own,
                                    bool Report, bool Reset) :
  DiagnosticsOverride(Diags, Own),
  m_Flags(Report | (Reset << 1)) {
}

DiagnosticsStore::~DiagnosticsStore() {
  if (m_Flags & kReport)
    Report();
  if (m_Flags & kReset)
    m_Diags.Reset(true);
}

void
DiagnosticsStore::HandleDiagnostic(clang::DiagnosticsEngine::Level Level,
                                   const clang::Diagnostic& Info) {
  m_Saved.emplace_back(clang::StoredDiagnostic(Level, Info));
  DiagnosticConsumer::HandleDiagnostic(Level, Info);
}

void
DiagnosticsStore::Report(bool DoReset) {
  // Don't wan't to report to ourself!
  ReplaceDiagnostics Tmp(*this);
  for (const clang::StoredDiagnostic& Diag : m_Saved)
    m_Diags.Report(Diag);
  if (DoReset)
    Reset();
}

} // namespace utils
} // namespace cling
