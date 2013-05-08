//===--- Reader.h - Input ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the abtract input interface.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_READER_H
#define TEXTINPUT_READER_H

#include "textinput/InputData.h"
#include <cstddef>

namespace textinput {
  class TextInputContext;

  using std::size_t;

  // Abstract input interface.
  class Reader {
  public:
    Reader(): fContext(0) {}
    virtual ~Reader();

    TextInputContext* GetContext() const { return fContext; }
    void SetContext(TextInputContext* C) { fContext = C; }

    virtual void GrabInputFocus() {}
    virtual void ReleaseInputFocus() {}

    virtual bool HavePendingInput(bool wait) = 0;
    virtual bool HaveBufferedInput() const { return false; }
    virtual bool ReadInput(size_t& nRead, InputData& in) = 0;
  private:
    TextInputContext* fContext; // Context object
  };
}

#endif // TEXTINPUT_READER_H
