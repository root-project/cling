//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S -Xclang -verify
// XFAIL: *
// REQUIRES: vanilla-cling

//All the currently failing stuff

.T Fail.h fwd_fail.h
#include "fwd_fail.h"
#include "Fail.h"
#include "FakeFwd.h" //Because we skip function templates having default function arguments
#include "FunctionTemplate.h"

//expected-no-diagnostics
.q
