//===--- TerminalDisplayWin.h - Output To Windows Console -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for writing to a Windows console
//  i.e. cmd.exe.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_TERMINALDISPLAYWIN_H
#define TEXTINPUT_TERMINALDISPLAYWIN_H

#include "textinput/TerminalDisplay.h"
#include <Windows.h>

namespace textinput {
  // Output to a Windows console or pipe.
  class TerminalDisplayWin: public TerminalDisplay {
  public:
    TerminalDisplayWin();
    ~TerminalDisplayWin();

    void HandleResizeEvent();
    void Clear() override;

    void Attach() override;
    void Detach() override;

  protected:
    void Move(Pos p) override;
    void MoveInternal(Pos p);
    void MoveUp(size_t nLines = 1) override;
    void MoveDown(size_t nLines = 1) override;
    void MoveLeft(size_t nCols = 1) override;
    void MoveRight(size_t nCols = 1) override;
    void MoveFront() override;
    void SetColor(char CIdx, const Color& C) override;
    void WriteRawString(const char* text, size_t len) override;

    void EraseToRight() override;
    void CheckCursorPos();

    void ShowError(const char* Where) const;

  private:
    size_t fStartLine; // line of current prompt in cmd.exe's buffer
    bool fIsAttached; // whether console is configured
    HANDLE fOut; // output handle
    DWORD fOldMode; // console configuration before grabbing
    DWORD fMyMode; // console configuration when active
    WORD fDefaultAttributes; // attributes to restore on destruction
    const UINT fOldCodePage; // saved codepage of console

    /// RAII object that temporarily enables VT sequences for the given console
    struct EnableVTProcessingRAII {
      HANDLE fConsole;
      DWORD fMode;
      EnableVTProcessingRAII(HANDLE con) : fConsole(con) {
        ::GetConsoleMode(fConsole, &fMode);
        ::SetConsoleMode(fConsole, fMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
      }
      ~EnableVTProcessingRAII() { ::SetConsoleMode(fConsole, fMode); }
    };
  };
}
#endif // TEXTINPUT_TERMINALDISPLAYWIN_H
