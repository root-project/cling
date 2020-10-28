//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -fno-rtti -Xclang -verify 2>&1 | FileCheck %s

// Test that user can override the DiagnosicsClient without error

#include <cling/Interpreter/Interpreter.h>
#include <cling/Utils/Diagnostics.h>
#include <cling/Utils/Output.h>
#include <clang/Frontend/CompilerInstance.h>

using namespace cling::utils;
DiagnosticsStore LC(gCling->getCI()->getDiagnostics(), false);
gCling->echo("error");

// When preprocessed out is supported, test that reporting works too.
// LC.Report();

for (const auto& D : LC) {
  cling::outs() << "STORED <" << D.getMessage() << ">\n";
}
// CHECK: STORED <use of undeclared identifier 'error'>

// expected-no-diagnostics
.q
