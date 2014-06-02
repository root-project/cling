//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S -Xclang -verify
// Test enumTest

enum class EC;
EC x=EC::A;
//expected-error {{}}
#include "Enum.h"
EC x=EC::A;
//expected-no-diagnostics
.q