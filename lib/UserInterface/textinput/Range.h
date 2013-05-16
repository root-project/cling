//===--- Range.h - From And Length ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines range-like structures, to hold what needs to be redrawn
//  or which parts of a text have been changed by the editor.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_RANGE_H
#define TEXTINPUT_RANGE_H

#include <cstddef>

namespace textinput {
  using std::size_t;

  // From and length; with whatever to update the prompt
  class Range {
  public:
    // Prompt is "$ "
    // Sometimes (extended input modes) followed by EditorPrompt, e.g.
    // "[bkwd''] "
    enum EPromptUpdate {
      kNoPromptUpdate = 0, // Don't redraw prompt
      kUpdatePrompt = 0x1, // Redraw the prompt
      kUpdateEditorPrompt = 0x2, // Redraw the editor prompt
      kUpdateAllPrompts = 0x3 // Redraw both prompts
    };

    Range(): fStart(0), fLength(0), fPromptUpdate(kNoPromptUpdate) {}
    Range(size_t pos): fStart(pos), fLength(1), fPromptUpdate(kNoPromptUpdate) {}
    Range(size_t pos, size_t length, EPromptUpdate PU = kNoPromptUpdate):
      fStart(pos), fLength(length), fPromptUpdate(PU) {}

    // Shortcuts to often-used values:
    static Range AllText() { return Range(0, (size_t)-1); }
    static Range AllWithPrompt() { return Range(0, (size_t)-1, kUpdateAllPrompts);}
    static Range Empty() { return Range(); }
    static size_t End() { return (size_t) -1; }

    Range& Extend(const Range& with);
    Range& Intersect(const Range& with);
    bool IsEmpty() const {
       return fLength == 0 && fPromptUpdate == kNoPromptUpdate; }
    void ExtendPromptUpdate(EPromptUpdate PU) {
      fPromptUpdate = (EPromptUpdate) ((int) fPromptUpdate | (int) PU); }

    static size_t PMax(size_t p1, size_t p2) { return (p1 > p2 ? p1 : p2); }
    static size_t PMin(size_t p1, size_t p2) { return (p1 < p2 ? p1 : p2); }

    size_t fStart; // Start of range, 0-based
    size_t fLength; // Length of range
    EPromptUpdate fPromptUpdate; // Which part of the prompt to update
  };

  class EditorRange {
  public:
    EditorRange() {}
    EditorRange(Range E, Range D): fEdit(E), fDisplay(D) {}
    Range fEdit;
    Range fDisplay;
  };
}
#endif // TEXTINPUT_RANGE_H
