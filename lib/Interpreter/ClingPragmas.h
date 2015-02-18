//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_PRAGMAS_H
#define CLING_PRAGMAS_H

namespace cling {
  class Interpreter;

  void addClingPragmas(Interpreter& interp);
} // namespace cling

#endif // CLING_PRAGMAS_H
