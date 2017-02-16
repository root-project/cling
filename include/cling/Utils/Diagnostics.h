//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author: Roman Zulak
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_UTILS_DIAGNOSTICS_H
#define CLING_UTILS_DIAGNOSTICS_H

#include "clang/Basic/Diagnostic.h"

namespace cling {
namespace utils {

    ///\brief Temporarily replace the DiagnosticConsumer in a DiagnosticsEngine
    ///
    class ReplaceDiagnostics {
    protected:
      clang::DiagnosticsEngine& m_Diags;
      clang::DiagnosticConsumer& m_PrevClient;
      const unsigned m_PrevOwn : 1;

    public:
      ///\brief ReplaceDiagnostics constructor
      ///
      ///\param[in] Diags - The DiagnosticsEngine instance to override
      ///\param[in] Replace - The DiagnosticConsumer to set as the new client
      ///\param[in] Own - Whether the DiagnosticsEngine owns that client
      ///
      ReplaceDiagnostics(clang::DiagnosticsEngine& Diags,
                         clang::DiagnosticConsumer& Replace, bool Own);

      ///\brief Temporarily restore the prior DiagnosticConsumer
      ///
      ///\param[in] Other - Which DiagnosticConsumer to restore to
      ///
      ReplaceDiagnostics(const ReplaceDiagnostics& Other)
          : ReplaceDiagnostics(Other.m_Diags, Other.m_PrevClient,
                               Other.m_PrevOwn) {}

      ~ReplaceDiagnostics() { m_Diags.setClient(&m_PrevClient, m_PrevOwn); }
    };

    ///\brief Temporarily override the DiagnosticConsumer in a DiagnosticsEngine
    /// Inherits from ReplaceDiagnostics so that forwarding can be done easily
    ///
    class DiagnosticsOverride : public clang::DiagnosticConsumer,
                                public ReplaceDiagnostics {
    public:
      ///\brief DiagnosticsOverride constructor
      ///
      ///\param[in] Diags - The DiagnosticsEngine instance to override
      ///\param[in] Own - Should DiagnosticsEngine should delete when done?
      ///
      DiagnosticsOverride(clang::DiagnosticsEngine& Diags, bool Own = false)
          : ReplaceDiagnostics(Diags, *this, Own) {}
    };

    ///\brief Store all of the errors sent to a DiagnosticsEngine
    ///
    class DiagnosticsStore : public DiagnosticsOverride {
      typedef std::vector<clang::StoredDiagnostic> Storage;

      // Bitfield first in the hopes it can be joined to ReplaceDiagnostics'
      const unsigned m_Flags : 2;
      Storage m_Saved;

      void HandleDiagnostic(clang::DiagnosticsEngine::Level Level,
                            const clang::Diagnostic& Info) override;

    public:
      ///\brief Store all further diagnostics
      ///
      ///\param[in] Diags - The DiagnosticsEngine instance to override
      ///\param[in] Own - Whether the DiagnosticsEngine owns this instance
      ///\param[in] Report - Report the collected error on destruction
      ///\param[in] Reset - Soft reset the DiagnosticsEngine on destruction
      ///
      DiagnosticsStore(clang::DiagnosticsEngine& Diags, bool Own,
                       bool Report = 1, bool Reset = 0);

      ~DiagnosticsStore();

      ///\brief Report the stored diagnostics to the previous DiagnosticConsumer
      ///
      ///\param[in] DoReset - Reset stored diagnostics after reporting
      ///
      void Report(bool DoReset = true);

      ///\brief Clear the stored diagnostics
      ///
      void Reset() {
        Storage().swap(m_Saved);
        DiagnosticsOverride::clear();
      }

      ///\brief STL interface to the stored diagnostics
      ///
      typedef Storage::value_type value_type;
      typedef Storage::iterator iterator;
      typedef Storage::const_iterator const_iterator;

      const_iterator begin() const { return m_Saved.begin(); }
      const_iterator end() const { return m_Saved.end(); }
      iterator begin() { return m_Saved.begin(); }
      iterator end() { return m_Saved.end(); }
      size_t size() const { return m_Saved.size(); }
      bool empty() const { return m_Saved.empty(); }
      const clang::StoredDiagnostic& operator[](int i) const {
        return m_Saved[i];
      }
    };

} // namespace utils
} // namespace cling

#endif // CLING_UTILS_DIAGNOSTICS_H
