//===--- TextInput.h - Main Interface ---------------------------*- C++ -*-===//
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

#ifndef TEXTINPUT_TEXTINPUT_H
#define TEXTINPUT_TEXTINPUT_H

#include <stddef.h>
#include <string>
#include <vector>

namespace textinput {
  class Colorizer;
  class Display;
  class EditorRange;
  class FunKey;
  class InputData;
  class Reader;
  class TextInputContext;
  class TabCompletion;

  // Main interface to textinput library.
  class TextInput {
  public:
    // State of input
    enum EReadResult {
      kRRNone, // uninitialized
      kRRReadEOLDelimiter, // end of line is entered, can take input
      kRRCharLimitReached, // SetMaxPendingCharsToRead() of input are read
      kRRNoMorePendingInput, // no input available
      kRREOF // end of file has been reached
    };

    TextInput(Reader& reader, Display& display,
              const char* histFile = 0);
    ~TextInput();

    // Getters
    const TextInputContext* GetContext() const { return fContext; }
    bool IsInputHidden() const { return fHidden; }

    size_t GetMaxPendingCharsToRead() const { return fMaxChars; }
    bool IsReadingAllPendingChars() const { return fMaxChars == (size_t) -1; }
    bool IsBlockingUntilEOL() const { return fMaxChars == 0; }

    // Setters
    void SetPrompt(const char* p);
    void HideInput(bool hidden = true) { fHidden = hidden; }
    void SetColorizer(Colorizer* c);
    void SetCompletion(TabCompletion* tc);
    void SetFunctionKeyHandler(FunKey* fc);

    void SetMaxPendingCharsToRead(size_t nMax) { fMaxChars = nMax; }
    void SetReadingAllPendingChars() { fMaxChars = (size_t) -1; }
    void SetBlockingUntilEOL() { fMaxChars = 0; }

    // Read interface
    EReadResult ReadInput();
    EReadResult GetReadState() const { return fLastReadResult; }
    char GetLastKey() const { return fLastKey; }
    const std::string& GetInput();
    void TakeInput(std::string& input); // Take and reset input
    bool AtEOL() const { return fLastReadResult == kRRReadEOLDelimiter || AtEOF(); }
    bool AtEOF() const { return fLastReadResult == kRREOF; }
    bool HavePendingInput() const;

    // Display interface
    void Redraw();
    void UpdateDisplay(const EditorRange& r);
    void DisplayInfo(const std::vector<std::string>& lines);
    void HandleResize();

    void GrabInputOutput() const;
    void ReleaseInputOutput() const;

    // History interface
    bool IsAutoHistAddEnabled() const { return fAutoHistAdd; }
    void EnableAutoHistAdd(bool enable = true) { fAutoHistAdd = enable; }
    void AddHistoryLine(const char* line);

  private:
    void EmitSignal(char c, EditorRange& r);
    void ProcessNewInput(const InputData& in, EditorRange& r);
    void DisplayNewInput(EditorRange& r, size_t& oldCursorPos);

    bool fHidden; // whether input should be shown
    bool fAutoHistAdd; // whether input should be added to history
    char fLastKey; // most recently read key
    size_t fMaxChars; // Num chars to read; 0 for blocking, -1 for all available
    EReadResult fLastReadResult; // current input state
    TextInputContext* fContext; // context object
    mutable bool fActive; // whether textinput is controlling input/output
    bool fNeedPromptRedraw; // whether the prompt should be redrawn on next attach
  };
}
#endif
