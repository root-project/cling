//===--- KeyBinding.cpp - Keys To InputData ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines how to convert raw input into normalized InputData.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#include "textinput/KeyBinding.h"

#include <ctype.h>

namespace textinput {
  KeyBinding::KeyBinding(): fEscPending(false), fEscCmdEnabled(false) {}
  KeyBinding::~KeyBinding() {}

  Editor::Command KeyBinding::ToCommand(InputData In) {
    // Convert InputData into a Command
    typedef Editor::Command C;
    bool HadEscPending = fEscPending;
    fEscPending = false;
    if (In.IsRaw()) {
      if (In.GetModifier() & InputData::kModCtrl) {
        return ToCommandCtrl(In.GetRaw(), HadEscPending);
      }

      if (HadEscPending) {
        return ToCommandEsc(In.GetRaw());
      }

      return C(In.GetRaw());
    }
    // else
    return ToCommandExtended(In.GetExtendedInput(), In.GetModifier(),
                             HadEscPending);
  }

  Editor::Command
  KeyBinding::ToCommandCtrl(char In,
                            bool HadEscPending) {
    // Control was pressed and In was hit. Convert to command.
    typedef Editor::Command C;
    switch (In) {
      case 'a' - 0x60: return C(Editor::kMoveFront);
      case 'b' - 0x60: return C(Editor::kMoveLeft);
      case 'c' - 0x60: return C(In, Editor::kCKControl);
      case 'd' - 0x60: return C(In, Editor::kCKControl);
      case 'e' - 0x60: return C(Editor::kMoveEnd);
      case 'f' - 0x60: return C(Editor::kMoveRight);
      case 'g' - 0x60: return C(Editor::kMoveRight);
      case 'h' - 0x60:
        if (HadEscPending) {
          return C(Editor::kCmdCutPrevWord);
        } else {
          return C(Editor::kCmdDelLeft);
        }
      case 'i' - 0x60: return C(Editor::kCmdComplete);
      case 'j' - 0x60: return C(Editor::kCmdEnter);
      case 'k' - 0x60: return C(Editor::kCmdCutToEnd);
      case 'l' - 0x60: return C(Editor::kCmdClearScreen);
      case 'm' - 0x60: return C(Editor::kCmdEnter);
      case 'n' - 0x60: return C(Editor::kCmdHistNewer);
      case 'o' - 0x60: return C(Editor::kCmdHistReplay);
      case 'p' - 0x60: return C(Editor::kCmdHistOlder);
      case 'q' - 0x60: return C(In, Editor::kCKError);
      case 'r' - 0x60: return C(Editor::kCmdReverseSearch);
      case 's' - 0x60: return C(Editor::kCmdForwardSearch);
      case 't' - 0x60: return C(Editor::kCmdSwapThisAndLeftThenMoveRight);
      case 'u' - 0x60: return C(Editor::kCmdCutToFront);
      case 'v' - 0x60: return C(In, Editor::kCKError);
      case 'w' - 0x60: return C(Editor::kCmdCutPrevWord);
      case 'x' - 0x60:
        // Jump to mark
        return C(In, Editor::kCKError);
      case 'y' - 0x60: return C(Editor::kCmdPaste);
      case 'z' - 0x60:
        return C(In, Editor::kCKControl);
      case 0x1f: return C(Editor::kCmdUndo);
      case 0x7f: // Backspace key (with Alt, or no modifier) on Unix, Del on MacOS
        if (HadEscPending) {
          return C(Editor::kCmdCutPrevWord);
        } else {
          return C(Editor::kCmdDelLeft);
        }
      default: return C(In, Editor::kCKError);
    }
    // Cannot reach:
    return C(In, Editor::kCKError);
  }

  Editor::Command
  KeyBinding::ToCommandEsc(char In) {
    // ESC was entered, followed by In. Convert to command.
    typedef Editor::Command C;
    switch (toupper(In)) {
      case 'B': return C(Editor::kMovePrevWord);
      case 'C': return C(Editor::kCmdToUpperMoveNextWord);
      case 'D': return C(Editor::kCmdCutNextWord);
      case 'F': return C(Editor::kMoveNextWord);
      case 'L': return C(Editor::kCmdWordToLower);
      case 'U': return C(Editor::kCmdWordToUpper);
      case 'i' - 0x60 /*TAB*/: return C(Editor::kCmdHistComplete);
      default: return C(In, Editor::kCKError);
    }
    // Cannot reach:
    return C(In, Editor::kCKError);
  }

  Editor::Command
  KeyBinding::ToCommandExtended(InputData::EExtendedInput EI,
                                unsigned char modifier,
                                bool HadEscPending) {
    // Convert extended input into the corresponding Command.
    typedef Editor::Command C;
    switch (EI) {
      case InputData::kEIUninitialized: return C(Editor::kCmdIgnore);
      case InputData::kEIHome: return C(Editor::kMoveFront);
      case InputData::kEIEnd: return C(Editor::kMoveEnd);
      case InputData::kEIUp: return C(Editor::kCmdHistOlder);
      case InputData::kEIDown: return C(Editor::kCmdHistNewer);
      case InputData::kEILeft:
        return (modifier & InputData::kModCtrl)
               ? C(Editor::kMovePrevWord) : C(Editor::kMoveLeft);
      case InputData::kEIRight:
        return (modifier & InputData::kModCtrl)
               ? C(Editor::kMoveNextWord) : C(Editor::kMoveRight);
      case InputData::kEIPgUp: return C(Editor::kCmdIgnore);
      case InputData::kEIPgDown: return C(Editor::kCmdIgnore);
      case InputData::kEIBackSpace:
        if (HadEscPending) {
          return C(Editor::kCmdCutPrevWord);
        } else {
          return C(Editor::kCmdDelLeft);
        }
      case InputData::kEIDel:
        if (HadEscPending) {
          return C(Editor::kCmdCutPrevWord);
        } else {
          return (modifier & InputData::kModCtrl)
                 ? C(Editor::kCmdCutNextWord) : C(Editor::kCmdDel);
        }
      case InputData::kEIIns: return C(Editor::kCmdToggleOverwriteMode);
      case InputData::kEITab: return C(Editor::kCmdComplete);
      case InputData::kEIEnter: return C(Editor::kCmdEnter);
      case InputData::kEIEsc:
        if (!fEscCmdEnabled) {
          // ESC can be CSI intro
          if (HadEscPending) {
            return C(Editor::kCmdEsc);
          }
          fEscPending = true;
          return C(Editor::kCmdIgnore);
        }
        return C(Editor::kCmdEsc);
      case InputData::kEIF1:
      case InputData::kEIF2:
      case InputData::kEIF3:
      case InputData::kEIF4:
      case InputData::kEIF5:
      case InputData::kEIF6:
      case InputData::kEIF7:
      case InputData::kEIF8:
      case InputData::kEIF9:
      case InputData::kEIF10:
      case InputData::kEIF11:
      case InputData::kEIF12:
        // pass to callback
      case InputData::kEIIgnore: return C(Editor::kCmdIgnore);
      default:  return C(Editor::kCmdIgnore);
    }
    // Cannot reach:
    return C(Editor::kCmdIgnore);
  }

}
