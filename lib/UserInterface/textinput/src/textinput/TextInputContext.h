//===--- TextInputContext.h - Object Holder ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the internal interface for TextInput's auxiliary
//  objects.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_TEXTINPUTCONFIG_H
#define TEXTINPUT_TEXTINPUTCONFIG_H

#include <vector>
#include "textinput/Text.h"

namespace textinput {
  class Colorizer;
  class Display;
  class Editor;
  class FunKey;
  class History;
  class KeyBinding;
  class Reader;
  class SignalHandler;
  class TabCompletion;
  class TextInput;

  // Context for textinput library. Collection of internal objects.
  class TextInputContext {
  private:
     TextInputContext(const TextInputContext&); // not implemented
     TextInputContext& operator=(const TextInputContext&); // not implemented
  public:
    TextInputContext(TextInput* ti, const char* histFile);
    ~TextInputContext();

    TextInput* GetTextInput() const { return fTextInput; }
    KeyBinding* GetKeyBinding() const { return fBind; }
    Editor* GetEditor() const { return fEdit; }
    SignalHandler* GetSignalHandler() const { return fSignal; }
    Colorizer* GetColorizer() const { return fColor; }
    History* GetHistory() const { return fHist; }
    TabCompletion* GetCompletion() const { return fTabCompletion; }
    FunKey* GetFunctionKeyHandler() const { return fFunKey; }
    void SetColorizer(Colorizer* C) { fColor = C; }
    void SetCompletion(TabCompletion* tc) { fTabCompletion = tc; }
    void SetFunctionKeyHandler(FunKey* fc) { fFunKey = fc; }

    const Text& GetPrompt() const { return fPrompt; }
    Text& GetPrompt() { return fPrompt; }
    void SetPrompt(const Text& P) { fPrompt = P; }

    const Text& GetLine() const { return fLine; }
    Text& GetLine() { return fLine; }
    void SetLine(const Text& T) { fLine = T; }

    size_t GetCursor() const { return fCursor; }
    void SetCursor(size_t C) { fCursor = C; }

    const std::vector<Display*>& GetDisplays() const { return fDisplays; }
    const std::vector<Reader*>& GetReaders() const { return fReaders; }
    TextInputContext& AddReader(Reader& R);
    TextInputContext& AddDisplay(Display& D);

  private:
    std::vector<Reader*> fReaders; // readers to use
    std::vector<Display*> fDisplays; // displays to write to
    TextInput* fTextInput; // Main textinput object
    KeyBinding* fBind; // key binding to use
    Editor* fEdit; // editor to use
    SignalHandler* fSignal; // signal handler to use
    Colorizer* fColor; // colorizer to use
    History* fHist; // history to use
    TabCompletion* fTabCompletion; // Tab completion handler
    FunKey* fFunKey; // Function key handler
    Text fPrompt; // current prompt
    Text fLine; // current input
    size_t fCursor; // input cursor position in fLine
  };
}
#endif
