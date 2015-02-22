//===--- TerminalDisplay.cpp - Output To Terminal ---------------*- C++ -*-===//
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

#include "textinput/TerminalDisplay.h"

#ifdef _WIN32
#include "textinput/TerminalDisplayWin.h"
#else
#include "textinput/TerminalDisplayUnix.h"
#endif

#include "textinput/TextInput.h"
#include "textinput/Color.h"
#include "textinput/Text.h"
#include "textinput/Editor.h"

namespace textinput {
  TerminalDisplay::~TerminalDisplay() {}

  TerminalDisplay*
  TerminalDisplay::Create() {
#ifdef _WIN32
    return new TerminalDisplayWin();
#else
    return new TerminalDisplayUnix();
#endif
  }

  void
  TerminalDisplay::NotifyTextChange(Range r) {
    if (!IsTTY()) return;
    Attach();
    WriteWrapped(r.fPromptUpdate,GetContext()->GetTextInput()->IsInputHidden(),
      r.fStart, r.fLength);
    Move(GetCursor());
  }

  void
  TerminalDisplay::NotifyCursorChange() {
    Attach();
    Move(GetCursor());
  }

  void
  TerminalDisplay::NotifyResetInput() {
    Attach();
    if (IsTTY()) {
      WriteRawString("\n", 1);
    }
    fWriteLen = 0;
    fWritePos = Pos();
  }

  void
  TerminalDisplay::NotifyError() {
    Attach();
    WriteRawString("\x07", 1);
  }

  void
  TerminalDisplay::DisplayInfo(const std::vector<std::string>& Options) {
    char infoColIdx = 0;
    if (GetContext()->GetColorizer()) {
       infoColIdx = GetContext()->GetColorizer()->GetInfoColor();
    }
    WriteRawString("\n", 1);
    for (size_t i = 0, n = Options.size(); i < n; ++i) {
      Text t(Options[i], infoColIdx);
      WriteWrappedElement(t, 0, 0, (size_t) -1);
      WriteRawString("\n", 1);
    }
    // Reset position
    Detach();
    Attach();
  }

  void
  TerminalDisplay::Detach() {
    fWritePos = Pos();
    fWriteLen = 0;
    if (GetContext()->GetColorizer()) {
      Color DefaultColor;
      GetContext()->GetColorizer()->GetColor(0, DefaultColor);
      SetColor(0, DefaultColor);
      // We can't tell whether the application will activate a different color:
      fPrevColor = -1;
    }
  }

  size_t
  TerminalDisplay::WriteWrappedElement(const Text& Element, size_t TextOffset,
                                       size_t WriteOffset, size_t Requested) {
    size_t Start = TextOffset;
    size_t Remaining = Requested;

    size_t Available = Element.length() - Start;
    if (Requested == (size_t) -1) {
      Requested = Available;
    }

    if (Available > 0) {
      if (Available < Remaining) {
        Remaining = Available;
      }

      while (Remaining > 0) {
        size_t numThisLine = Remaining;

        // How much can this line hold?
        size_t numToEOL = GetWidth() - ((Start + WriteOffset) % GetWidth());
        if (!numToEOL) {
          MoveDown();
          ++fWritePos.fLine;
          MoveFront();
          fWritePos.fCol = 0;
          numToEOL = GetWidth();
        }
        if (numThisLine > numToEOL) {
          numThisLine = numToEOL;
        }

        if (GetContext()->GetColorizer()) {
          // We only write same-color chunks; how long is it?
          const std::vector<char>& Colors = Element.GetColors();
          char ThisColor = Colors[Start];
          size_t numSameColor = 1;
          while (numSameColor < numThisLine
                 && ThisColor == Colors[Start + numSameColor])
            ++numSameColor;
          numThisLine = numSameColor;

          if (ThisColor != fPrevColor) {
            Color C;
            GetContext()->GetColorizer()->GetColor(ThisColor, C);
            SetColor(ThisColor, C);
            fPrevColor = ThisColor;
          }
        }

        WriteRawString(Element.GetText().c_str() + Start, numThisLine);
        fWritePos = IndexToPos(PosToIndex(fWritePos) + numThisLine);
        if (numThisLine == numToEOL) {
          ActOnEOL();
        }

        Start += numThisLine;
        Remaining -= numThisLine;
      }
    }

    if (Requested == Available) {
      size_t VisL = fWriteLen / GetWidth();
      size_t Wrote = WriteOffset + TextOffset + Requested;
      size_t WroteL = Wrote / GetWidth();
      size_t NumToEOL = GetWidth() - (Wrote % GetWidth());
      if (fWriteLen > Wrote && NumToEOL > 0) {
        // Wrote less and not at EOL
        EraseToRight();
      }
      if (WroteL < VisL) {
        Pos prevWC = GetCursor();
        MoveFront();
        fWritePos.fCol = 0;
        for (size_t l = WroteL + 1; l <= VisL; ++l) {
          MoveDown();
          ++fWritePos.fLine;
          EraseToRight();
        }
        Move(prevWC);
      }
    }
    return Remaining;
  }

  size_t
  TerminalDisplay::WriteWrapped(Range::EPromptUpdate PromptUpdate, bool hidden,
                                size_t Offset, size_t Requested /* = -1*/) {
    Attach();

    const Text& Prompt = GetContext()->GetPrompt();
    size_t PromptLen = GetContext()->GetPrompt().length();
    const Text& EditPrompt = GetContext()->GetEditor()->GetEditorPrompt();
    size_t EditorPromptLen = EditPrompt.length();

    if (!IsTTY()) {
       PromptLen = 0;
       EditorPromptLen = 0;
       PromptUpdate = Range::kNoPromptUpdate;
    }

    if (PromptUpdate & Range::kUpdatePrompt) {
      // Writing from front means we write the prompt, too
      Move(Pos());
      WriteWrappedElement(Prompt, 0, 0, PromptLen);
    }
    if (PromptUpdate != Range::kNoPromptUpdate) {
      // Any prompt update means we'll have to re-write the editor prompt
      Move(IndexToPos(PromptLen));
      if (EditorPromptLen) {
        WriteWrappedElement(EditPrompt, 0, PromptLen, EditorPromptLen);
      }
      // Any prompt update means we'll have to re-write the text
      Offset = 0;
      Requested = (size_t) -1;
    }
    Move(IndexToPos(PromptLen + EditorPromptLen + Offset));

    size_t avail = 0;
    if (hidden) {
      Text hide(std::string(GetContext()->GetLine().length(), '*'), 0);
      avail = WriteWrappedElement(hide, Offset,
                                  PromptLen + EditorPromptLen, Requested);
    } else {
      avail = WriteWrappedElement(GetContext()->GetLine(), Offset,
                                       PromptLen + EditorPromptLen, Requested);
    }
    fWriteLen = PromptLen + EditorPromptLen + GetContext()->GetLine().length();
    return avail;
  }

  void
  TerminalDisplay::Move(Pos p) {
    Attach();
    if (fWritePos == p) return;
    if (fWritePos.fLine > p.fLine) {
      MoveUp(fWritePos.fLine - p.fLine);
      fWritePos.fLine -= fWritePos.fLine - p.fLine;
    } else if (fWritePos.fLine < p.fLine) {
      MoveDown(p.fLine - fWritePos.fLine);
      fWritePos.fLine += p.fLine - fWritePos.fLine;
    }

    if (p.fCol == 0) {
      MoveFront();
      fWritePos.fCol = 0;
    } else if (fWritePos.fCol > p.fCol) {
      MoveLeft(fWritePos.fCol - p.fCol);
      fWritePos.fCol -= fWritePos.fCol - p.fCol;
    } else if (p.fCol > fWritePos.fCol) {
      MoveRight(p.fCol - fWritePos.fCol);
      fWritePos.fCol += p.fCol - fWritePos.fCol;
    }
  }
}
