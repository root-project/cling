//===--- TerminalReaderWin.h - Input From Windows Console -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for reading from Window's cmd.exe console.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_STREAMREADERWIN_H
#define TEXTINPUT_STREAMREADERWIN_H

#include "textinput/StreamReader.h"
#include <Windows.h>

namespace textinput {
  // Windows console and pipe input
  class StreamReaderWin: public StreamReader {
  public:
    StreamReaderWin();
    ~StreamReaderWin();

    void GrabInputFocus();
    void ReleaseInputFocus();

    bool HavePendingInput(bool wait);
    bool ReadInput(size_t& nRead, InputData& in);

  private:
    void HandleError(const char* Where) const;
    void HandleKeyEvent(unsigned char C, InputData& in);

    bool fHaveInputFocus; // whether the console is configured
    bool fIsConsole; // whether the input is a console or file
    HANDLE fIn; // input handle
    DWORD fOldMode; // configuration before grabbing input device
    DWORD fMyMode; // configuration while active
  };
}

#endif // TEXTINPUT_STREAMREADERWIN_H
