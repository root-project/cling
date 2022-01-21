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

  ////////////////////////////////////////////////////////////////////////////////
  /// Notify the display that the text has been changed in range r.
  /// Rewrite the display in range r and move back to the cursor.
  ///
  /// \param[in] r Range to write out the text for.
  void
  TerminalDisplay::NotifyTextChange(Range r) {
    if (!IsTTY()) return;
    Attach();
    WriteWrapped(r.fPromptUpdate, GetContext()->GetTextInput()->IsInputMasked(),
      r.fStart, r.fLength);
    Move(GetCursor());
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Notify the display that the cursor has been changed. Move to the cursor.
  void
  TerminalDisplay::NotifyCursorChange() {
    Attach();
    Move(GetCursor());
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Notify the display that the input has been taken.
  /// Move to the next line, reset written length and position.
  void
  TerminalDisplay::NotifyResetInput() {
    Attach();
    if (IsTTY()) {
      WriteRawString("\n", 1);
    }
    fWriteLen = 0;
    fWritePos = Pos();
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Notify the display that there has been an error.
  /// Write out the BEL character.
  void
  TerminalDisplay::NotifyError() {
    Attach();
    WriteRawString("\x07", 1);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Display an informational message at the prompt.
  /// Acts like a pop-up. Used e.g. for tab-completion.
  ///
  /// \param[in] Options options to write out
  void
  TerminalDisplay::DisplayInfo(const std::vector<std::string>& Options) {
    char infoColIdx = 0;
    if (GetContext()->GetColorizer()) {
       infoColIdx = GetContext()->GetColorizer()->GetInfoColor();
    }
    WriteRawString("\n", 1);
    for (size_t i = 0, n = Options.size(); i < n; ++i) {
      Text t(Options[i], infoColIdx);
      WriteWrappedTextPart(t, 0, 0, (size_t) -1);
      WriteRawString("\n", 1);
    }
    // Reset position
    Detach();
    Attach();
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Detach from the abstract display by resetting the position
  /// and written text length. If Colorizer is present, reset the color too.
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

  ////////////////////////////////////////////////////////////////////////////////
  /// Write out wrapped text to the display. Used in WriteWrapped and DisplayInfo
  ///
  /// \param[in] text text to write out
  /// \param[in] TextOffset where to begin writing out text from
  /// \param[in] WriteOffset where to begin writing out text at the display
  /// \param[in] NumRequested number of text characters requested for output
  size_t
  TerminalDisplay::WriteWrappedTextPart(const Text &text, size_t TextOffset,
                                        size_t WriteOffset, size_t NumRequested) {
    size_t Start = TextOffset;
    size_t NumRemaining = NumRequested; // optimistic

    size_t NumAvailable = text.length() - Start;
    if (NumRequested == (size_t) -1) { // requested max available
      NumRequested = NumAvailable;
    }

    // If we have some text available for output
    if (NumAvailable > 0) {
      // If we don't have enough to output NumRemaining, output only what's available
      if (NumAvailable < NumRemaining) {
        NumRemaining = NumAvailable;
      }

      while (NumRemaining > 0) {
        // How much can this line hold?
        size_t numToEOL = GetWidth() - ((Start + WriteOffset) % GetWidth());
        if (numToEOL == 0) { // we are at EOL, move down
          MoveDown();
          ++fWritePos.fLine;
          MoveFront();
          fWritePos.fCol = 0;
          numToEOL = GetWidth();
        }

        // How much of our text can we fit in this line?
        size_t numThisLine;
        if (NumRemaining > numToEOL) {
          numThisLine = numToEOL;
        } else {
          numThisLine = NumRemaining;
        }

        // If there is a Colorizer, we only write same-colored chunks.
        // How long is the current chunk? Adjust numThisLine.
        if (GetContext()->GetColorizer()) {
          const std::vector<char>& Colors = text.GetColors();
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

        // Write out the line and update the write position
        WriteRawString(text.GetText().c_str() + Start, numThisLine);
        fWritePos = IndexToPos(PosToIndex(fWritePos) + numThisLine);
        if (numThisLine == numToEOL) { // If we hit EOL, wrap around
          ActOnEOL();
        }

        Start += numThisLine;
        NumRemaining -= numThisLine;
      }
    }

    // If we have processed the characters we have requested
    if (NumRequested == NumAvailable) {
      size_t NumPrevLines = fWriteLen / GetWidth();
      size_t LenWrote = WriteOffset + TextOffset + NumRequested;
      size_t NumWroteLines = LenWrote / GetWidth();
      size_t NumToEOL = GetWidth() - (LenWrote % GetWidth());
      if (LenWrote < fWriteLen && NumToEOL > 0) {
        // If we wrote less than previously and not at EOL
        // Erase the rest of the current line
        EraseToRight();
      }
      if (NumWroteLines < NumPrevLines) {
        // If we wrote less lines than previously,
        // erase the surplus previous lines
        Pos prevWC = GetCursor();
        MoveFront();
        fWritePos.fCol = 0;
        for (size_t l = NumWroteLines + 1; l <= NumPrevLines; ++l) {
          MoveDown();
          ++fWritePos.fLine;
          EraseToRight();
        }
        Move(prevWC);
      }
    }
    return NumRemaining;
  }

  size_t
  TerminalDisplay::WriteWrapped(Range::EPromptUpdate PromptUpdate, bool masked,
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

    // If updating prompt, write the main prompt first (e.g. [cling]$)
    if (PromptUpdate & Range::kUpdatePrompt) {
      // Writing from front means we write the prompt, too
      Move(Pos());
      WriteWrappedTextPart(Prompt, 0, 0, PromptLen);
    }
    // If updating any prompt
    if (PromptUpdate != Range::kNoPromptUpdate) {
      // Any prompt update means we'll have to re-write the editor prompt
      Move(IndexToPos(PromptLen));
      if (EditorPromptLen) {
        WriteWrappedTextPart(EditPrompt, 0, PromptLen, EditorPromptLen);
      }
      // Any prompt update means we'll have to re-write the text
      Offset = 0;
      Requested = (size_t) -1;
    }
    Move(IndexToPos(PromptLen + EditorPromptLen + Offset));

    size_t avail = 0;
    if (masked) {
      Text mask(std::string(GetContext()->GetLine().length(), '*'), 0);
      avail = WriteWrappedTextPart(mask, Offset,
                                   PromptLen + EditorPromptLen, Requested);
    } else {
      avail = WriteWrappedTextPart(GetContext()->GetLine(), Offset,
                                   PromptLen + EditorPromptLen, Requested);
    }
    fWriteLen = PromptLen + EditorPromptLen + GetContext()->GetLine().length();
    return avail;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Move the cursor to the required position.
  ///
  /// \param[in] p position to move to
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
