//===--- KeyBinding.h - Keys To InputData -----------------------*- C++ -*-===//
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

#ifndef TEXTINPUT_KEYBINDING_H
#define TEXTINPUT_KEYBINDING_H

#include "textinput/Editor.h"
#include "textinput/InputData.h"

namespace textinput {
  // Convert InputData to Editor::Command
  class KeyBinding {
  public:
    KeyBinding();
    ~KeyBinding();

    Editor::Command ToCommand(InputData In);
    void EnableEscCmd(bool enable) {
      // Whether "Esc" is allowed to mean something by itself.
      // Is can be misinterpreted as the start of a CSI terminal
      // sequence; use this to disambiguate.
      fEscCmdEnabled = enable; fEscPending = false; }
    bool IsEscCommandEnabled() const {
      // Whether "Esc" has a meaning by itself, e.g.
      // during history search
      return fEscCmdEnabled;
    }
    bool IsEscPending() const { return fEscPending; }

  private:
    Editor::Command ToCommandCtrl(char In, bool HadEscPending);
    Editor::Command ToCommandEsc(char In);
    Editor::Command ToCommandExtended(InputData::EExtendedInput EI,
                                      unsigned char modifier,
                                      bool HadEscPending);
    bool fEscPending; // Dangling ESC is waiting to be processed
    bool fEscCmdEnabled; // Single ESC has a meaning
  };
}
#endif // TEXTINPUT_KEYBINDING_H
