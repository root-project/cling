//===--- TerminalReader.cpp - Input From Terminal ---------------*- C++ -*-===//
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

#include "textinput/StreamReader.h"

#ifdef WIN32
# include "textinput/StreamReaderWin.h"
#else
# include "textinput/StreamReaderUnix.h"
#endif

namespace textinput {
  StreamReader::~StreamReader() {}

  StreamReader*
  StreamReader::Create() {
#ifdef WIN32
     return new StreamReaderWin();
#else
     return new StreamReaderUnix();
#endif
  }
}
