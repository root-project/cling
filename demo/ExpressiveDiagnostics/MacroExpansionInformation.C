//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The demo shows if there was an error in a macro, the expansion location
// macro definition is shown
// Author: Vassil Vassilev <vvasilev@cern.ch>

#define MAX(A, B) ((A) > (B) ? (A) : (B))

struct A {
  int Y;
};

void MacroExpansionInformation () {
  int X = 1;
  A* SomeA = new A();
  X = MAX(X, *SomeA);
}
