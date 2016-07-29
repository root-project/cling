//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify
// XFAIL: powerpc64
//This file checks an if statement for null prt dereference.

#include <stdlib.h>

int* p = 0;

if (*p) { exit(1); } // expected-warning {{null passed to a callee that requires a non-null argument}}

if (true) { *p; exit(1); } // expected-warning {{null passed to a callee that requires a non-null argument}}

if (false) {} else { *p; exit(1); } // expected-warning {{null passed to a callee that requires a non-null argument}}

.q
