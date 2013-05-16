//===--- TerminalDisplay.h - Output To Terminal -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the abstract interface for writing to a terminal.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_TERMINALDISPLAY_H
#define TEXTINPUT_TERMINALDISPLAY_H

#include <cstddef>                      // for size_t
#include <string>                       // for string
#include <vector>                       // for vector
#include "textinput/Display.h"
#include "textinput/Editor.h"
#include "textinput/Range.h"            // for Range
#include "textinput/Text.h"             // for Text
#include "textinput/TextInputContext.h"

namespace textinput {
  class Color;

  // Base class for output to a terminal.
  class TerminalDisplay: public Display {
  public:
    ~TerminalDisplay();
    static TerminalDisplay* Create();

    void NotifyTextChange(Range r);
    void NotifyCursorChange();
    void NotifyResetInput();
    void NotifyError();
    void Detach();
    void DisplayInfo(const std::vector<std::string>& Options);
    bool IsTTY() const { return fIsTTY; }

  protected:
    TerminalDisplay(bool isTTY):
      fIsTTY(isTTY), fWidth(80), fWriteLen(0), fPrevColor(-1) {}
    void SetIsTTY(bool isTTY) { fIsTTY = isTTY; }
    Pos GetCursor() const {
      // Collect the different prompts and the text cursor to calculate
      // the cursor position in the terminal.
      size_t idx = GetContext()->GetCursor();
      idx += GetContext()->GetPrompt().length();
      idx += GetContext()->GetEditor()->GetEditorPrompt().length();
      return IndexToPos(idx);
    }
    Pos IndexToPos(size_t idx) const { return Pos(idx % fWidth, idx / fWidth); }
    size_t PosToIndex(const Pos& pos) const {
      // Convert a x|y position to an index.
      return pos.fCol + pos.fLine * fWidth; }
    size_t GetWidth() const { return fWidth; }
    void SetWidth(size_t width) { fWidth = width; }

    virtual void Move(Pos p);
    virtual void MoveUp(size_t nLines = 1) = 0;
    virtual void MoveDown(size_t nLines = 1) = 0;
    virtual void MoveLeft(size_t nCols = 1) = 0;
    virtual void MoveRight(size_t nCols = 1) = 0;
    virtual void MoveFront() = 0;
    size_t WriteWrapped(Range::EPromptUpdate PromptUpdate, bool hidden,
                        size_t offset, size_t len = (size_t)-1);
    size_t WriteWrappedElement(const Text& what, size_t TextOffset,
                               size_t WriteOffset, size_t Requested);
    virtual void SetColor(char CIdx, const Color& C) = 0;
    virtual void WriteRawString(const char* text, size_t len) = 0;
    virtual void ActOnEOL() {}

    virtual void EraseToRight() = 0;

  protected:
    bool fIsTTY; // whether this is a terminal or redirected
    size_t fWidth; // Width of the terminal in character columns
    size_t fWriteLen; // Last char of output written.
    Pos fWritePos; // Current position of writing (temporarily != cursor)
    char fPrevColor; // currently configured color
  };
}
#endif // TEXTINPUT_TERMINALDISPLAY_H
