//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_VALUEPRINTER_H
#define CLING_VALUEPRINTER_H

namespace llvm {
  class raw_ostream;
}

namespace cling {
  class Value;

  namespace valuePrinterInternal {
    void printValue_Default(llvm::raw_ostream& o, const Value& V);
    void printType_Default(llvm::raw_ostream& o, const Value& V);
  } // end namespace valuePrinterInternal
}

#endif // CLING_VALUEPRINTER_H
