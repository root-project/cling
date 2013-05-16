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

#include "textinput/Range.h"

namespace textinput {
  Range&
  Range::Extend(const Range& with) {
    if (IsEmpty()) {
      *this = with;
      return *this;
    }
    if (with.IsEmpty()) return *this;

    size_t wEnd = with.fStart + with.fLength;
    if (with.fLength == (size_t) -1) wEnd = (size_t) -1;
    size_t end = fStart + fLength;
    if (fLength == (size_t) -1) end = (size_t) -1;

    fStart = PMin(fStart, with.fStart);
    end = PMax(end, wEnd);
    if (end == (size_t) -1) {
      fLength = (size_t) -1;
    } else {
      fLength = end - fStart;
    }
    fPromptUpdate = (EPromptUpdate) (fPromptUpdate | with.fPromptUpdate);
    return *this;
  }

  Range&
  Range::Intersect(const Range& with) {
    if (IsEmpty()) {
      return *this;
    }
    if (with.IsEmpty()) {
      *this = Empty();
      return *this;
    }
    size_t wEnd = with.fStart + with.fLength;
    if (with.fLength == (size_t) -1) wEnd = (size_t) -1;
    size_t end = fStart + fLength;
    if (fLength == (size_t) -1) end = (size_t) -1;

    fStart = PMax(fStart, with.fStart);
    end = PMin(end, wEnd);
    if (end == (size_t) -1) {
      fLength = (size_t) -1;
    } else {
      fLength = end - fStart;
    }
    return *this;
  }
}
