//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Timur Pocheptsov <Timur.Pocheptsov@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_DISPLAY_H
#define CLING_DISPLAY_H

#include <string>

namespace llvm {
  class raw_ostream;
}

namespace cling {
class Interpreter;

void DisplayClass(llvm::raw_ostream &stream,
                  const Interpreter *interpreter, const char *className,
                  bool verbose);

void DisplayNamespaces(llvm::raw_ostream &stream, const Interpreter *interpreter);

void DisplayGlobals(llvm::raw_ostream &stream, const Interpreter *interpreter);
void DisplayGlobal(llvm::raw_ostream &stream, const Interpreter *interpreter,
                   const std::string &name);

void DisplayTypedefs(llvm::raw_ostream &stream, const Interpreter *interpreter);
void DisplayTypedef(llvm::raw_ostream &stream, const Interpreter *interpreter,
                    const std::string &name);

}

#endif
