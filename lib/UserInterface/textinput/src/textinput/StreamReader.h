//===--- TerminalReader.h - Input From Terminal -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface reading from a terminal.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_STREAMREADER_H
#define TEXTINPUT_STREAMREADER_H
#include "textinput/Reader.h"

namespace textinput {
  // Base class for input from a terminal or file descriptor.
  class StreamReader: public Reader {
  public:
    ~StreamReader();
    static StreamReader* Create();

  protected:
    StreamReader() {}
  };
}
#endif // TEXTINPUT_STREAMREADER_H
