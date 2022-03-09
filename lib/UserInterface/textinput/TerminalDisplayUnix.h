//===--- TerminalDisplayUnix.h - Output To UNIX Terminal --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for writing to a UNIX terminal. It tries to
//  support all "common" terminal types.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_TERMINALDISPLAYUNIX_H
#define TEXTINPUT_TERMINALDISPLAYUNIX_H

#include <cstddef>
#include "textinput/TerminalDisplay.h"

namespace textinput {
  class Color;

  // Output to tty / pipe / file.
  class TerminalDisplayUnix: public TerminalDisplay {
  public:
    TerminalDisplayUnix();
    ~TerminalDisplayUnix();

    void HandleResizeSignal();
    void Clear() override;

    void Attach() override;
    void Detach() override;

  protected:
    void MoveUp(size_t nLines = 1) override;
    void MoveDown(size_t nLines = 1) override;
    void MoveLeft(size_t nCols = 1) override;
    void MoveRight(size_t nCols = 1) override;
    void MoveInternal(char What, size_t n);
    void MoveFront() override;
    void SetColor(char CIdx, const Color& C) override;
    void WriteRawString(const char* text, size_t len) override;
    void ActOnEOL() override;
    void EraseToRight() override;
    int GetClosestColorIdx256(const Color& C);
    int GetClosestColorIdx16(const Color& C);

  private:
    bool fIsAttached; // whether tty is configured
    size_t fNColors; // number of colors supported by output
    int fOutputID; // Prompt output file descriptor
  };
}
#endif // TEXTINPUT_TERMINALDISPLAYUNIX_H
