//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S -Xclang -verify
// Test FwdPrinterTest

// Test similar to ROOT-7037
// Equivalent to parsing the dictionary preamble
.T Def2b.h fwd_Def2b.h
#include "fwd_Def2b.h"
// Then doing the autoparsing
#include "Def2b.h"
A<int> ai2;
// And then do both a second time with a different template instantiation.
.T Def2c.h fwd_Def2c.h
#include "fwd_Def2c.h"
#include "Def2c.h"

// In some implementations the AutoloadingVisitor was stripping the default
// template parameter value from the class template definition leading to
// compilation error at this next line:
A<float> af2;

.T Def2.h fwd_Def2.h
#include "fwd_Def2.h"
#include "Def2.h"

.T Enum.h fwd_enums.h
#include "fwd_enums.h"
#include "Enum.h"

//expected-no-diagnostics
.q
