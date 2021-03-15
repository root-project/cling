//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S -Xclang -verify 2>&1

#pragma cling load("P0.h", "P1.h","P2.h") //expected-error {{expected P0.h to be a library, but it is not. If this is a source file, use `#include "P0.h"`}}

.q
