//===--- TerminalReaderUnix.h - Input From UNIX Terminal --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface reading from a UNIX terminal. It tries to
//  support all common terminal types.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_STREAMREADERUNIX_H
#define TEXTINPUT_STREAMREADERUNIX_H

#include "textinput/StreamReader.h"
#include <cstddef>
#include <queue>

namespace textinput {
  class InputData;

  // Input from a tty, file descriptor, or pipe
  class StreamReaderUnix: public StreamReader {
  public:
    StreamReaderUnix();
    ~StreamReaderUnix();

    void GrabInputFocus() override;
    void ReleaseInputFocus() override;

    bool HavePendingInput(bool wait) override;
    bool HaveBufferedInput() const override { return !fReadAheadBuffer.empty(); }
    bool ReadInput(size_t& nRead, InputData& in) override;

    bool IsFromTTY() override { return fIsTTY; }
  private:
    int ReadRawCharacter();
    bool ProcessCSI(InputData& in);

    bool fHaveInputFocus; // whether we configured the tty
    bool fIsTTY; // whether input FD is a tty
    std::queue<char> fReadAheadBuffer; // input chars we read too much (CSI)
  };
}

#endif // TEXTINPUT_STREAMREADERUNIX_H
