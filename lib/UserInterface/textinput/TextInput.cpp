//===--- TextInput.cpp - Main Interface -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the main interface for the TextInput library.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#include "textinput/TextInput.h"
#include "textinput/Color.h"
#include "textinput/Display.h"
#include "textinput/Editor.h"
#include "textinput/History.h"
#include "textinput/InputData.h"
#include "textinput/KeyBinding.h"
#include "textinput/Range.h"
#include "textinput/Reader.h"
#include "textinput/SignalHandler.h"
#include "textinput/Text.h"
#include "textinput/TextInputContext.h"

#include <algorithm>
#include <functional>

namespace textinput {
  TextInput::TextInput(Reader& reader, Display& display,
                       const char* HistFile /* = 0 */):
  fMasked(false),
  fAutoHistAdd(true),
  fLastKey(0),
  fMaxChars(0),
  fLastReadResult(kRRNone),
  fActive(false),
  fNeedPromptRedraw(false)
  {
    fContext = new TextInputContext(this, HistFile);
    fContext->AddDisplay(display);
    fContext->AddReader(reader);
  }

  TextInput::~TextInput() {
    delete fContext;
  }

  void
  TextInput::TakeInput(std::string &input, bool force) {
    if (!force && fLastReadResult != kRRReadEOLDelimiter
        && fLastReadResult != kRREOF) {
      input.clear();
      return;
    }
    input = fContext->GetLine().GetText();
    while (!input.empty() && input[input.length() - 1] == 13) {
      input.erase(input.length() - 1);
    }
    fContext->GetEditor()->ResetText();

    // Signal displays that the input got taken.
    std::for_each(fContext->GetDisplays().begin(), fContext->GetDisplays().end(),
             [](Display *D) { return D->NotifyResetInput(); });

    ReleaseInputOutput();

    if (force || fLastReadResult == kRRReadEOLDelimiter) {
      // Input has been taken, we can continue reading.
      fLastReadResult = kRRNone;
      // We will have to redraw the prompt, even if unchanged.
      fNeedPromptRedraw = true;
    } else {
      fLastReadResult = kRREOF;
    }
  }

  bool
  TextInput::HavePendingInput() const {
    if (!fActive) {
      GrabInputOutput();
    }
    for (std::vector<Reader*>::const_iterator iR = fContext->GetReaders().begin(),
         iE = fContext->GetReaders().end(); iR != iE; ++iR) {
      if ((*iR)->HavePendingInput(false))
        return true;
    }
    return false;
  }

  TextInput::EReadResult
  TextInput::ReadInput() {
    // Read more input.

    // Don't read if we are at the end; force call to TakeInput().
    if (fLastReadResult == kRRReadEOLDelimiter
        || fLastReadResult == kRREOF)
      return fLastReadResult;

    if (fLastReadResult == kRRNone) {
      GrabInputOutput();
      // Asking for input, so prompt must have been drawn already
      UpdateDisplay(EditorRange(Range::AllText(), Range::AllText()));
    }

    size_t nRead = 0;
    size_t nMax = GetMaxPendingCharsToRead();
    if (nMax == 0) nMax = (size_t) -1; // aka "all"
    InputData in;
    EditorRange R;
    size_t OldCursorPos = fContext->GetCursor();
    // Allow the reader to select() if we want a single line, and we want it
    // only from one reader
    bool waitForInput = IsBlockingUntilEOL()
      && fContext->GetReaders().size() == 1;
    for (std::vector<Reader*>::const_iterator iR
           = fContext->GetReaders().begin(),
         iE = fContext->GetReaders().end();
         iR != iE && nRead < nMax; ++iR) {
      while ((IsBlockingUntilEOL() && (fLastReadResult == kRRNone))
             || (nRead < nMax && (*iR)->HavePendingInput(waitForInput))
             || (*iR)->HaveBufferedInput()) {
        if ((*iR)->ReadInput(nRead, in)) {
          ProcessNewInput(in, R);
          DisplayNewInput(R, OldCursorPos);
          if (fLastReadResult == kRREOF
              || fLastReadResult == kRRReadEOLDelimiter)
            break;
        }
      }
    }

    if (fLastReadResult == kRRNone) {
      if (nRead == nMax) {
        fLastReadResult = kRRCharLimitReached;
      } else {
        fLastReadResult = kRRNoMorePendingInput;
      }
    }
    return fLastReadResult;
  }

  void
  TextInput::ProcessNewInput(const InputData& in, EditorRange& R) {
    // in was read, process it.
    fLastKey = in.GetRaw(); // rough approximation
    Editor::Command Cmd = fContext->GetKeyBinding()->ToCommand(in);

    // Translate Ctrl-D to delete if input line is not empty, as GNU readline does.
    if (!fContext->GetLine().empty() && Cmd.isCtrlD())
      Cmd = Editor::Command(Editor::kCmdDel);

    if (Cmd.GetKind() == Editor::kCKControl
        && (Cmd.GetChar() == 3 || Cmd.GetChar() == 26)) {
      // If there are modifications in the queue, process them now.
      UpdateDisplay(R);
      HandleControl(Cmd.GetChar(), R);
    } else if (Cmd.GetKind() == Editor::kCKCommand
               && Cmd.GetCommandID() == Editor::kCmdWindowResize) {
      std::for_each(fContext->GetDisplays().begin(),
                    fContext->GetDisplays().end(),
                    [](Display *D) { return D->NotifyWindowChange(); });
    } else if (Cmd.isCtrlD()) {
      fContext->SetLine(".q"), Redraw();
      fLastReadResult = kRREOF;
      return;
    } else {
      if (!in.IsRaw() && in.GetExtendedInput() == InputData::kEIEOF) {
        fLastReadResult = kRREOF;
        return;
      } else {
        Editor::EProcessResult Res = fContext->GetEditor()->Process(Cmd, R);
        if (Res == Editor::kPRError) {
          // Signal displays that an error has occurred.
          std::for_each(fContext->GetDisplays().begin(),
                        fContext->GetDisplays().end(),
                        [](Display *D) { return D->NotifyError(); });
        } else if (Cmd.GetKind() == Editor::kCKCommand
                   && (Cmd.GetCommandID() == Editor::kCmdEnter ||
                       Cmd.GetCommandID() == Editor::kCmdHistReplay)) {
          fLastReadResult = kRRReadEOLDelimiter;
          return;
        }
      }
    }
  }

  void
  TextInput::DisplayNewInput(EditorRange& R, size_t& oldCursorPos) {
    // Display what has been entered.
    if (fContext->GetColorizer() && oldCursorPos != fContext->GetCursor()) {
       fContext->GetColorizer()->ProcessCursorChange(fContext->GetCursor(),
                                                     fContext->GetLine(),
                                                     R.fDisplay);
    }

    UpdateDisplay(R);

    if (oldCursorPos != fContext->GetCursor()) {
      std::for_each(fContext->GetDisplays().begin(), fContext->GetDisplays().end(),
                    [](Display *D) { return D->NotifyCursorChange(); });
    }

    oldCursorPos = fContext->GetCursor();
  }

  void
  TextInput::Redraw() {
    // Attach and redraw.
    GrabInputOutput();
    fNeedPromptRedraw = false;
    UpdateDisplay(EditorRange(Range::AllText(), Range::AllWithPrompt()));
  }

  void
  TextInput::UpdateDisplay(const EditorRange& R) {
    // Update changed ranges if attached.
    if (!fActive) {
      return;
    }
    EditorRange ColModR(R);
    if (!R.fDisplay.IsEmpty() && fContext->GetColorizer()) {
      fContext->GetColorizer()->ProcessTextChange(ColModR, fContext->GetLine());
    }
    if (fNeedPromptRedraw) {
       ColModR.fDisplay.fPromptUpdate = Range::kUpdateAllPrompts;
       fNeedPromptRedraw = false;
    }
    if (!ColModR.fDisplay.IsEmpty()
        || ColModR.fDisplay.fPromptUpdate != Range::kNoPromptUpdate) {
      std::for_each(fContext->GetDisplays().begin(), fContext->GetDisplays().end(),
                    [ColModR](Display *D) { return D->NotifyTextChange(ColModR.fDisplay); });
    }
  }

  void
  TextInput::HandleControl(char C, EditorRange& R) {
    if (C == 3) { // Control+C
      std::string input = fContext->GetLine().GetText();
      size_t length = input.size();
      fContext->SetLine(input + "^C");
      UpdateDisplay(EditorRange(Range(length), Range::AllText()));
      TakeInput(input, true);
    } else if (C == 26) { // Control+Z
      ReleaseInputOutput();
      SignalHandler* Signal = fContext->GetSignalHandler();
      Signal->EmitCtrlZ();
    }

    GrabInputOutput();

    R.fDisplay = Range::AllWithPrompt();
    fNeedPromptRedraw = false;
    // Immediate refresh.
    std::for_each(fContext->GetDisplays().begin(), fContext->GetDisplays().end(),
                  [R](Display *D) { return D->NotifyTextChange(R.fDisplay); });
    // Empty range.
    R.fDisplay = Range::Empty();

  }

  void
  TextInput::SetPrompt(const char *P) {
    fContext->SetPrompt(P);
    if (fContext->GetColorizer()) {
      fContext->GetColorizer()->ProcessPromptChange(fContext->GetPrompt());
    }
    if (!fActive) {
       fNeedPromptRedraw = true;
       return;
    }
    fNeedPromptRedraw = false;
    std::for_each(fContext->GetDisplays().begin(),
                  fContext->GetDisplays().end(),
                  [](Display *D) { return D->NotifyTextChange(Range::AllWithPrompt()); });
  }

  void
  TextInput::SetColorizer(Colorizer* c) {
    fContext->SetColorizer(c);
  }

  void
  TextInput::SetCompletion(TabCompletion* tc) {
    // Set the completion handler.
    fContext->SetCompletion(tc);
  }
  void
  TextInput::SetFunctionKeyHandler(FunKey* fc) {
    fContext->SetFunctionKeyHandler(fc);
  }

  void
  TextInput::GrabInputOutput() const {
    if (fActive) return;
    // Signal readers that we are about to read.
    std::for_each(fContext->GetReaders().begin(), fContext->GetReaders().end(),
                  [](Reader *R) { return R->GrabInputFocus(); });
    // Signal displays that we are about to display.
    std::for_each(fContext->GetDisplays().begin(), fContext->GetDisplays().end(),
                  [](Display *D) { return D->Attach(); });
    fActive = true;
  }

  void
  TextInput::ReleaseInputOutput() const {
    // Signal readers that we are done reading.
    if (!fActive) return;
    std::for_each(fContext->GetReaders().begin(), fContext->GetReaders().end(),
                  [](Reader *R) { return R->ReleaseInputFocus(); });

    // Signal displays that we are done displaying.
    std::for_each(fContext->GetDisplays().begin(), fContext->GetDisplays().end(),
                  [](Display *D) { return D->Detach(); });

    fActive = false;
  }

  void
  TextInput::DisplayInfo(const std::vector<std::string>& lines) {
    // Display an informational message at the prompt. Acts like
    // a pop-up. Used e.g. for tab-completion.

    // foreach fails to build the reference in GCC 4.1.
    // Iterate manually instead.
    for (std::vector<Display*>::const_iterator i = fContext->GetDisplays().begin(),
         e = fContext->GetDisplays().end(); i != e; ++i) {
      (*i)->DisplayInfo(lines);
    }
  }

  void
  TextInput::HandleResize() {
    // Resize signal was emitted, tell the displays.
    std::for_each(fContext->GetDisplays().begin(), fContext->GetDisplays().end(),
                  [](Display *D) { return D->NotifyWindowChange(); });
  }

  void
  TextInput::AddHistoryLine(const char* line) {
    if (!line) return;
    std::string sLine(line);
    while (!sLine.empty()
           && (sLine[sLine.length() - 1] == '\n'
               || sLine[sLine.length() - 1] == '\r')) {
      sLine.erase(sLine.length() - 1);
    }
    if (!sLine.empty()) {
       fContext->GetHistory()->AddLine(sLine);
    }
  }
}
