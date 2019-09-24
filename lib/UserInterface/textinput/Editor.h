//===--- Editor.h - Output Of Text ------------------------------*- C++ -*-===//
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

#ifndef TEXTINPUT_EDITOR_H
#define TEXTINPUT_EDITOR_H

#include <deque>
#include <cstddef>                      // for size_t
#include <string>                       // for string
#include <utility>                      // for pair
#include "textinput/Text.h"
#include "textinput/Range.h"

namespace textinput {
  class TextInputContext;

  // Text manipulation.
  class Editor {
  public:
    // Kind of editor command
    enum ECommandKind {
      kCKError, // general unhappiness
      kCKCommand, // a real editing command
      kCKChar, // character input
      kCKMove, // cursor movement
      kCKControl // control character not translated to kCKCommand
    };

    // Editing command
    enum ECommandID {
      kCmdDel, // delete the char at the cursor position
      kCmdDelLeft, // delete char left of cursor, move there (backspace)

      // Cutting means: erase and store in paste buf
      kCmdCutToEnd, // cut text from cursor to end of line
      kCmdCutToFront, // cut text from cursor to front
      kCmdCutPrevWord, // cut text from cursor to beginning of word
      kCmdCutNextWord, // cut text from cursor to end of word
      kCmdPaste, // past from all subsequent, same direction cut

      kCmdSwapThisAndLeftThenMoveRight,

      kCmdWordToUpper,
      kCmdWordToLower,
      kCmdToUpperMoveNextWord,

      kCmdReverseSearch,
      kCmdHistOlder,
      kCmdHistNewer,
      kCmdHistReplay,
      kCmdHistComplete,

      kCmdComplete, // TAB - fires a callback.

      kCmd_END_TEXT_MODIFYING_CMDS,

      kCmdEnter, // Usually end of line

      kCmdInsertMode,
      kCmdOverwiteMode,
      kCmdToggleOverwriteMode,

      kCmdClearScreen,
      kCmdWindowResize,

      kCmdUndo,

      kCmdEsc,
      kCmdIgnore // ignore this command, e.g. because it was already processed
    };

    enum EMoveID {
      kMoveLeft,
      kMoveRight,
      //kMoveUp, - that's history
      //kMoveDown, - that's history
      kMoveFront,
      kMoveEnd,
      kMoveNextWord,
      kMovePrevWord
    };

    // Whether the editing command was successful
    enum EProcessResult {
      kPRError,
      kPRSuccess
    };

    // A compound editing command.
    class Command {
    public:
      Command(ECommandID C): fKind(kCKCommand), fCmd(C) {}
      Command(EMoveID M): fKind(kCKMove), fMove(M) {}
      Command(char C, ECommandKind k = kCKChar): fKind(k), fChar(C) {}

      ECommandKind GetKind() const { return fKind; }

      ECommandID GetCommandID() const { return fCmd;}
      EMoveID GetMoveID() const { return fMove;}
      char GetChar() const { return fChar;}

      bool isCtrlD() const { return fKind == kCKControl
                                    && (fChar == 'd'-0x60); }
    private:
      ECommandKind fKind; // discriminator for union
      union {
        ECommandID fCmd; // editor command value
        EMoveID fMove; // move value
        char fChar; // character input value
      };
    };

    Editor(TextInputContext* C):
      fContext(C), fCurHistEntry((size_t)-1), fReplayHistEntry((size_t)-1),
      fMode(kInputMode), fOverwrite(false), fCutDirection(0) {}
    ~Editor() {}

    Range ResetText();
    EProcessResult Process(Command Cmd, EditorRange& R);

    const Text& GetEditorPrompt() const { return fEditorPrompt; }
    void SetEditorPrompt(const Text& EP) { fEditorPrompt = EP; }
    void CancelSpecialInputMode(Range& DisplayR);
    void CancelAndRevertSpecialInputMode(EditorRange& R);

  private:
    EProcessResult ProcessChar(char C, EditorRange& R);
    EProcessResult ProcessMove(EMoveID M, EditorRange& R);
    EProcessResult ProcessCommand(ECommandID M, EditorRange& R);
    size_t FindWordBoundary(int Direction);
    void PushUndo();

    void AddToPasteBuf(int Dir, const std::string& T);
    void AddToPasteBuf(int Dir, char T);
    void ClearPasteBuf() { fCutDirection = 0; }
    void SetReverseHistSearchPrompt(Range& RDisplay);
    bool UpdateHistSearch(EditorRange& R);

    // The editor can be in special modes, e.g. when searching
    // in history.
    enum EEditMode {
      kInputMode, // regular input mode
      kHistSearchMode, // searching in history
      kNumEditModes
    };

    TextInputContext* fContext; // Context object
    Text fEditorPrompt; // for special modes, e.g. reverse search
    std::string fLineNotInHist; // current input line, not pushed to hist yet
    std::string fPasteBuf; // cut strings that can be pasted
    std::string fSearch; // for backward hist search
    size_t fCurHistEntry; // the current line stems from a hist entry, -1 if not
    size_t fReplayHistEntry; // set next line to this hist entry, kCmdHistReplay
    EEditMode fMode; // current input mode
    bool fOverwrite; // Insert of overwrite
    int fCutDirection; // cutting forward or wackward - change clears pastbuf
    std::deque<std::pair<Text /*Line*/, size_t /*Cursor*/> > fUndoBuf; // undos
  };
}
#endif // TEXTINPUT_EDITOR_H
