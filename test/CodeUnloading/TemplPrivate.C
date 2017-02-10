//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -DTEST_LITE=1 -I%S --noruntime -Xclang -verify 2>&1
// RUN: cat %s | %cling -I%S --noruntime -Xclang -verify 2>&1
// Test privateTemplateFunctionParam

.storeState "Initial"
#include "TemplPrivate.h"
.undo
.compareState "Initial"

// expected-no-diagnostics
.q
