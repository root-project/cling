//===--- Display.h - Output Of Text -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the abtract base for text output.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_DISPLAY_H
#define TEXTINPUT_DISPLAY_H

#include <string>
#include <vector>
#include "textinput/Range.h"

namespace textinput {
  class Text;
  class TextInputContext;

  // Abstract interface for displaying text.
  class Display {
  public:
    // Position with 0-based line and column.
    // The line is 0 for each new input line, i.e.
    // it's relative to the most recent prompt.
    struct Pos {
      Pos() : fCol(0), fLine(0) {}
      Pos(size_t col, size_t line): fCol(col), fLine(line) {}

      bool operator==(const Pos& O) const {
        return fCol == O.fCol && fLine == O.fLine; }

      size_t fCol;
      size_t fLine;
    };

    Display(): fContext(nullptr) {}
    virtual ~Display();

    const TextInputContext* GetContext() const { return fContext; }
    void SetContext(TextInputContext* C) { fContext = C; }

    /// If is a TTY, clear the terminal screen
    virtual void Clear() {}
    virtual void Redraw() { NotifyTextChange(Range::AllWithPrompt()); }

    virtual void NotifyTextChange(Range r) = 0; // Update the displayed text
    virtual void NotifyCursorChange() {} // Move the cursor
    virtual void NotifyResetInput() {} // The input was "taken", next prompt
    virtual void NotifyError() {} // An error occurred
    virtual void NotifyWindowChange() {} // The window's dimensions changed
    virtual void DisplayInfo(const std::vector<std::string>& Options) = 0;//Info
    virtual void Attach() {} // Take control e.g. of the terminal
    virtual void Detach() {} // Allow others to control terminal's parameters

  private:
    const TextInputContext* fContext; // Context object
  };
}
#endif // TEXTINPUT_DISPLAY_H
