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

struct Point {
  float x;
  float y;
};

void FixItHints() {
  struct Point origin = { x: 0.0, y: 0.0 };
  floot p;
}
