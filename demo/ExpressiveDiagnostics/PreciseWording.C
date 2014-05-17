//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The demo shows the Fix-it hints that try to guess what the user had meant
// when they did the error
// Author: Vassil Vassilev <vvasilev@cern.ch>

struct A {
  int X;
};

int PreciseWording () {
  A SomeA;
  int y = *SomeA.X;
}
