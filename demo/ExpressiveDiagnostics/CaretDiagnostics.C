//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The demo shows that cling (clang) can report even the column number of the
// error and emit caret
// Author: Vassil Vassilev <vvasilev@cern.ch>

#include <stdio.h>

void CaretDiagnostics() {
  int i = 5;
  printf("%.*d\n",i);
}
