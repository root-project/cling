//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Javier Lopez-Gomez <j.lopez@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------
#ifndef CLING_RUNTIME_OPTIONS_H
#define CLING_RUNTIME_OPTIONS_H

namespace cling {
  namespace runtime {
    /// \brief Interpreter configuration bits that can be changed at run-time
    /// by the user, e.g. to enable/disable extensions.
    struct RuntimeOptions {
      RuntimeOptions() : AllowRedefinition(0) {}

      /// \brief Allow the user to redefine entities (requests enabling the
      /// `DefinitionShadower` AST transformer).
      bool AllowRedefinition : 1;
    };

  } // end namespace runtime
} // end namespace cling

#endif // CLING_RUNTIME_OPTIONS_H
