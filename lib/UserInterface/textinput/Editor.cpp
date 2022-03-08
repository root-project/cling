//===--- Editor.cpp - Output Of Text ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the text manipulation ("editing") interface.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//


#include "textinput/Editor.h"

#include <cctype>
#include <vector>
#include "textinput/Callbacks.h"
#include "textinput/Display.h"
#include "textinput/History.h"
#include "textinput/KeyBinding.h"
#include "textinput/StreamReaderUnix.h"
#include "textinput/TextInput.h"
#include "textinput/TextInputContext.h"

namespace textinput {

  // Functions to find first/last non alphanumeric ("word-boundaries")
  size_t find_first_non_alnum(const std::string &str,
                              std::string::size_type index = 0) {
    bool atleast_one_alnum = false;
    std::string::size_type len = str.length();
    for(; index < len; ++index) {
      const char c = str[index];
      bool is_alpha = isalnum(c) || c == '_';
      if (is_alpha) atleast_one_alnum = true;
      else if (atleast_one_alnum) return index;
    }
    return std::string::npos;
  }

  size_t find_last_non_alnum(const std::string &str,
                             std::string::size_type index = std::string::npos) {
    std::string::size_type len = str.length();
    if (index == std::string::npos) index = len - 1;
    bool atleast_one_alnum = false;
    for(; index != std::string::npos; --index) {
      const char c = str[index];
      bool is_alpha = isalnum(c) || c == '_';
      if (is_alpha) atleast_one_alnum = true;
      else if (atleast_one_alnum) return index;
    }
    return std::string::npos;
  }

  Editor::EProcessResult
  Editor::Process(Command cmd, EditorRange& R) {
    switch (cmd.GetKind()) {
      case kCKChar:
        return ProcessChar(cmd.GetChar(), R);
      case kCKMove:
        return ProcessMove(cmd.GetMoveID(), R);
      case kCKCommand:
        return ProcessCommand(cmd.GetCommandID(), R);
      case kCKControl:
      case kCKError:
        return kPRError;
    }
    return kPRError;
  }

  Range
  Editor::ResetText() {
    bool addToHist = !fContext->GetLine().empty()
      && !fContext->GetTextInput()->IsInputMasked()
      && fContext->GetTextInput()->IsAutoHistAddEnabled();
    if (addToHist) {
      fContext->GetHistory()->AddLine(fContext->GetLine().GetText());
      if (fReplayHistEntry != (size_t) -1) {
        // Added a line, thus renumber
        ++fReplayHistEntry;
      }
    }
    Range R(0, fContext->GetLine().length());
    fContext->GetLine().clear();
    fContext->SetCursor(0);
    ClearPasteBuf();
    fSearch.clear();
    CancelSpecialInputMode(R);
    if (fReplayHistEntry != (size_t) -1) {
      --fReplayHistEntry; // intentional overflow to -1
      fContext->SetLine(fContext->GetHistory()->GetLine(fReplayHistEntry));
    }
    return R;
  }

  void
  Editor::SetReverseHistSearchPrompt(Range& RDisplay) {
    std::string P("[bkw'");
    SetEditorPrompt(Text(P + fSearch + "'] "));
    RDisplay.ExtendPromptUpdate(Range::kUpdateEditorPrompt);
  }

  bool
  Editor::UpdateHistSearch(EditorRange& R) {
    History* Hist = fContext->GetHistory();
    Text& Line = fContext->GetLine();
    size_t NewHistEntry = (size_t) -1;
    if (fSearch.empty()) {
      NewHistEntry = 0;
    } else {
      size_t startAt = fCurHistEntry;
      if (startAt == (size_t) -1) {
        startAt = 0;
      }
      for (size_t i = startAt, n = Hist->GetSize(); i < n; ++i) {
        if (Hist->GetLine(i).find(fSearch) != std::string::npos) {
          NewHistEntry = i;
          break;
        }
      }
    }
    if (NewHistEntry != (size_t) -1) {
      // No, even if they are unchanged: we might have
      // subsequent ^R updates triggered by faking a different
      // fCurHistEntry.
      // if (NewHistEntry != fCurHistEntry) {
      fCurHistEntry = NewHistEntry;
      Line = Hist->GetLine(fCurHistEntry);
      R.fEdit.Extend(Range::AllText());
      R.fDisplay.Extend(Range::AllText());
      // Resets mode, thus can't call:
      // ProcessMove(kMoveEnd, R);
      fContext->SetCursor(Line.length());
      return true;
    }

    fCurHistEntry = (size_t) -1;
    return false;
  }

  void
  Editor::CancelSpecialInputMode(Range& DisplayR) {
    // Stop incremental history search, leaving text at the
    // history line currently selected.
    if (fMode == kInputMode) return;
    fContext->GetKeyBinding()->EnableEscCmd(false);
    SetEditorPrompt(Text());
    DisplayR.ExtendPromptUpdate(Range::kUpdateEditorPrompt);
    fMode = kInputMode;
  }

  void
  Editor::CancelAndRevertSpecialInputMode(EditorRange& R) {
    // Stop incremental history search, reset text to what it was
    // before search started.
    if (fMode == kInputMode) return;
    CancelSpecialInputMode(R.fDisplay);
    // Original line should be top of undo buffer.
    ProcessCommand(kCmdUndo, R);
  }

  Editor::EProcessResult
  Editor::ProcessChar(char C, EditorRange& R) {
    if (C < 32) return kPRError;

    if (fMode == kHistSearchMode) {
      fSearch += C;
      SetReverseHistSearchPrompt(R.fDisplay);
      if (UpdateHistSearch(R)) return kPRSuccess;
      return kPRError;
    }

    PushUndo();
    ClearPasteBuf();

    Text& Line = fContext->GetLine();
    size_t Cursor = fContext->GetCursor();

    if (fOverwrite) {
      if (Cursor < Line.length()) {
        Line[Cursor] = C;
      } else {
        Line += C;
      }
      R.fEdit.Extend(Range(Cursor));
      R.fDisplay.Extend(Range(Cursor));
    } else {
      Line.insert(Cursor, C);
      R.fEdit.Extend(Range(Cursor));
      R.fDisplay.Extend(Range(Cursor, Range::End()));
      fContext->SetCursor(Cursor + 1);
    }
    return kPRSuccess;
  }

  Editor::EProcessResult
  Editor::ProcessMove(EMoveID M, EditorRange &R) {
    if (fMode == kHistSearchMode) {
       if (M == kMoveRight) {
          // ^G, i.e. cancel hist search and revert original line.
          CancelAndRevertSpecialInputMode(R);
          return kPRSuccess;
       }
    }

    ClearPasteBuf();
    CancelSpecialInputMode(R.fDisplay);

    size_t Cursor = fContext->GetCursor();
    size_t LineLen = fContext->GetLine().length();

    switch (M) {
      case kMoveEnd: fContext->SetCursor(LineLen); return kPRSuccess;
      case kMoveFront: fContext->SetCursor(0); return kPRSuccess;
      case kMoveRight:
        if (Cursor < LineLen) {
          fContext->SetCursor(Cursor + 1);
          return kPRSuccess;
        } else {
          return kPRError;
        }
      case kMoveLeft:
        if (Cursor > 0) {
          fContext->SetCursor(Cursor - 1);
          return kPRSuccess;
        } else {
          return kPRError;
        }
      case kMoveNextWord:
      case kMovePrevWord:
        fContext->SetCursor(FindWordBoundary(M == kMoveNextWord ? 1 : -1));
        return kPRSuccess;
    }
    return kPRError;
  }

  Editor::EProcessResult
  Editor::ProcessCommand(ECommandID M, EditorRange &R) {
    if (M < kCmd_END_TEXT_MODIFYING_CMDS) {
      PushUndo();
    }
    if (fMode == kHistSearchMode) {
      if (M == kCmdDelLeft) {
        if (fSearch.empty()) return kPRError;
        fSearch.erase(fSearch.length() - 1);
        SetReverseHistSearchPrompt(R.fDisplay);
        if (UpdateHistSearch(R)) return kPRSuccess;
        return kPRError;
      } else if (M == kCmdReverseSearch) {
        // Search again. Move to older hist entry:
        size_t prevHistEntry = fCurHistEntry;
        // intentional overflow from -1 to 0:
        if (fCurHistEntry + 1 >= fContext->GetHistory()->GetSize()) {
          return kPRError;
        }
        if (fCurHistEntry == (size_t)-1) {
          fCurHistEntry = 0;
        } else {
          ++fCurHistEntry;
        }
        if (UpdateHistSearch(R)) return kPRSuccess;
        fCurHistEntry = prevHistEntry;
        return kPRError;
      } else {
        CancelSpecialInputMode(R.fDisplay);
        return kPRError;
      }
    }

    Text& Line = fContext->GetLine();
    size_t Cursor = fContext->GetCursor();
    History* Hist = fContext->GetHistory();

    switch (M) {
      case kCmdIgnore:
        return kPRSuccess;
      case kCmdEnter:
        fReplayHistEntry = (size_t) -1;
        fCurHistEntry = (size_t) -1;
        CancelSpecialInputMode(R.fDisplay);
        return kPRSuccess;
      case kCmdDelLeft:
        if (Cursor == 0) return kPRError;
        fContext->SetCursor(--Cursor);
        // intentional fallthrough:
      case kCmdDel:
        if (Cursor == Line.length()) return kPRError;
        AddToPasteBuf(M == kCmdDel ? 1 : -1, Line[Cursor]);
        Line.erase(Cursor);
        R.fEdit.Extend(Range(Cursor));
        R.fDisplay.Extend(Range(Cursor, Range::End()));
        return kPRSuccess;
      case kCmdCutToEnd:
        AddToPasteBuf(1, Line.GetText().c_str() + Cursor);
        Line.erase(Cursor, Line.length() - Cursor);
        R.fEdit.Extend(Range(Cursor));
        R.fDisplay.Extend(Range(Cursor, Range::End()));
        return kPRSuccess;
      case kCmdCutNextWord:
      {
        size_t posWord = FindWordBoundary(1);
        AddToPasteBuf(1, Line.GetText().substr(Cursor, posWord - Cursor));
        R.fEdit.Extend(Range(Cursor, posWord));
        R.fDisplay.Extend(Range(Cursor, Range::End()));
        Line.erase(Cursor, posWord - Cursor);
        return kPRSuccess;
      }
      case kCmdCutPrevWord:
      {
        size_t posWord = FindWordBoundary(-1);
        AddToPasteBuf(-1, Line.GetText().substr(posWord, Cursor - posWord));
        R.fEdit.Extend(Range(posWord, Cursor));
        R.fDisplay.Extend(Range(posWord, Range::End()));
        Line.erase(posWord, Cursor - posWord);
        fContext->SetCursor(posWord);
        return kPRSuccess;
      }
      case kCmdToggleOverwriteMode:
        fOverwrite = !fOverwrite;
        return kPRSuccess;
      case kCmdInsertMode:
        fOverwrite = false;
        return kPRSuccess;
      case kCmdOverwiteMode:
        fOverwrite = true;
        return kPRSuccess;
      case kCmdCutToFront:
        R.fEdit.Extend(Range(0, Cursor));
        R.fDisplay.Extend(Range::AllText());
        AddToPasteBuf(-1, Line.GetText().substr(0, Cursor));
        Line.erase(0, Cursor);
        fContext->SetCursor(0);
        return kPRSuccess;
      case kCmdPaste:
      {
        size_t PasteLen = fPasteBuf.length();
        R.fEdit.Extend(Range(Cursor, PasteLen));
        R.fDisplay.Extend(Range(Cursor, Range::End()));
        Line.insert(Cursor, fPasteBuf);
        fContext->SetCursor(Cursor + PasteLen);
        ClearPasteBuf();
        return kPRSuccess;
      }
      case kCmdSwapThisAndLeftThenMoveRight:
      {
        if (Cursor < 1) return kPRError;
        R.fEdit.Extend(Range(Cursor - 1, Cursor));
        R.fDisplay.Extend(Range(Cursor - 1, Cursor));
        char tmp = Line.GetText()[Cursor];
        Line[Cursor] = Line[Cursor - 1];
        Line[Cursor - 1] = tmp;
        // optional:
        ProcessMove(kMoveRight, R);
        return kPRSuccess;
      }
      case kCmdToUpperMoveNextWord:
      {
        if (Cursor >= Line.length()) return kPRError;
        Line[Cursor] = toupper(Line[Cursor]);
        R.fEdit.Extend(Range(Cursor));
        R.fDisplay.Extend(Range(Cursor));
        ProcessMove(kMoveNextWord, R);
        return kPRSuccess;
      }
      case kCmdWordToLower:
      case kCmdWordToUpper:
      {
        size_t posWord = FindWordBoundary(1);
        if (M == kCmdWordToUpper) {
          for (size_t i = Cursor; i < posWord; ++i) {
            Line[Cursor] =  toupper(Line[Cursor]);
          }
        } else {
          for (size_t i = Cursor; i < posWord; ++i) {
            Line[Cursor] =  tolower(Line[Cursor]);
          }
        }
        R.fEdit.Extend(Range(Cursor, posWord - Cursor));
        R.fDisplay.Extend(Range(Cursor, posWord - Cursor));
        fContext->SetCursor(posWord);
        return kPRSuccess;
      }
      case kCmdUndo:
        Line = fUndoBuf.back().first;
        fContext->SetCursor(fUndoBuf.back().second);
        fUndoBuf.pop_back();
        R.fEdit.Extend(Range::AllText());
        R.fDisplay.Extend(Range::AllText());
        return kPRSuccess;
      case kCmdHistNewer:
        // already newest?
        if (fCurHistEntry == (size_t)-1) {
          // not a history line ("newer" doesn't mean anything)?
          return kPRError;
        }
        if (fCurHistEntry == 0) {
          Hist->ModifyLine(fCurHistEntry, Line.GetText().c_str());
          Line = fLineNotInHist;
          fLineNotInHist.clear();
          fCurHistEntry = (size_t)-1; // not in hist
        } else {
          --fCurHistEntry;
          Line = Hist->GetLine(fCurHistEntry);
        }
        R.fEdit.Extend(Range::AllText());
        R.fDisplay.Extend(Range::AllText());
        ProcessMove(kMoveEnd, R);
        return kPRSuccess;
      case kCmdHistOlder:
        // intentional overflow from -1 to 0:
        if (fCurHistEntry + 1 >= Hist->GetSize()) {
          return kPRError;
        }
        if (fCurHistEntry == (size_t)-1) {
          fLineNotInHist = Line.GetText();
          fCurHistEntry = 0;
        } else {
          Hist->ModifyLine(fCurHistEntry, Line.GetText().c_str());
          ++fCurHistEntry;
        }
        Line = Hist->GetLine(fCurHistEntry);
        R.fEdit.Extend(Range::AllText());
        R.fDisplay.Extend(Range::AllText());
        ProcessMove(kMoveEnd, R);
        return kPRSuccess;
      case kCmdReverseSearch:
        PushUndo();
        fMode = kHistSearchMode;
        fSearch.clear();
        SetReverseHistSearchPrompt(R.fDisplay);
        fContext->GetKeyBinding()->EnableEscCmd(true);
        if (UpdateHistSearch(R)) return kPRSuccess;
        return kPRError;
      case kCmdHistReplay:
        if (fCurHistEntry == (size_t) -1) return kPRError;
        fReplayHistEntry = fCurHistEntry;
        return kPRSuccess;
      case kCmdClearScreen:
        for (auto *D : fContext->GetDisplays()) {
          D->Clear();
          D->Redraw();
        }
        return kPRSuccess;
      case kCmd_END_TEXT_MODIFYING_CMDS:
        return kPRError;
      case kCmdEsc:
        // Already done for all commands:
        //CancelSpecialInputMode(R);
        return kPRSuccess;
      case kCmdComplete:
      {
        // Completion happens below current input.
        ProcessMove(kMoveEnd, R);
        std::vector<std::string> completions;
        TabCompletion* tc = fContext->GetCompletion();
        Reader* reader = fContext->GetReaders()[0];
        StreamReaderUnix* streamReader = (StreamReaderUnix*)(reader);
        if (!streamReader->IsFromTTY()) return kPRSuccess;
        if (tc) {
          bool ret = tc->Complete(Line, Cursor, R, completions);
          if (ret) {
            if (!completions.empty()) {
              fContext->GetTextInput()->DisplayInfo(completions);
              R.fDisplay.Extend(Range::AllWithPrompt());
            }
            fContext->SetCursor(Cursor);
            return kPRSuccess;
          }
        }
        return kPRError;
      }
      case kCmdWindowResize:
        fContext->GetTextInput()->HandleResize();
        return kPRSuccess;
      case kCmdHistComplete:
        // Not handled yet, todo.
        return kPRError;
    }
    return kPRError;
  }

  size_t
  Editor::FindWordBoundary(int Direction) {

    const Text& Line = fContext->GetLine();
    size_t Cursor = fContext->GetCursor();

    if (Direction < 0 && Cursor < 2) return 0;

    size_t ret = Direction > 0 ?
      find_first_non_alnum(Line.GetText(), Cursor + 1)
    : find_last_non_alnum(Line.GetText(), Cursor - 2);

    if (ret == std::string::npos) {
      if (Direction > 0) return Line.length();
      else return 0;
    }

    if (Direction < 0)
      ret += 1;

    if (ret == std::string::npos) {
      if (Direction > 0) return Line.length();
      else return 0;
    }
    return ret;
  }

  void
  Editor::AddToPasteBuf(int Dir, std::string const &T) {
    if (fCutDirection == Dir) {
      if (Dir < 0) {
        fPasteBuf = T + fPasteBuf;
      } else {
        fPasteBuf += T;
      }
    } else {
      fCutDirection = Dir;
      fPasteBuf = T;
    }
  }

  void
  Editor::AddToPasteBuf(int Dir, char T) {
    if (fCutDirection == Dir) {
      if (Dir < 0) {
        fPasteBuf = std::string(1, T) + fPasteBuf;
      } else {
        fPasteBuf += T;
      }
    } else {
      fCutDirection = Dir;
      fPasteBuf = T;
    }
  }

  void
  Editor::PushUndo() {
    static const size_t MaxUndoBufSize = 100;
    if (fUndoBuf.size() > MaxUndoBufSize) {
      fUndoBuf.pop_front();
    }
    fUndoBuf.push_back(std::make_pair(fContext->GetLine(),
                                      fContext->GetCursor()));
  }

   // Pin vtables:
   TabCompletion::~TabCompletion() {}
   FunKey::~FunKey() {}
}
