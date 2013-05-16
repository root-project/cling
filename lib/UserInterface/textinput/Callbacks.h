//===--- Callbacks.h - Hook Registration ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for acting on certain input.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_CALLBACKS_H
#define TEXTINPUT_CALLBACKS_H

#include <vector>
#include <string>

namespace textinput {
  class Text;
  class EditorRange;

  class TabCompletion {
  public:
    // Returns false on error
    virtual bool Complete(Text& Line /*in+out*/,
                          size_t& Cursor /*in+out*/,
                          EditorRange& R /*out*/,
                          std::vector<std::string>& DisplayCompletions /*out*/)
    = 0;
    virtual ~TabCompletion();
  };

  class FunKey {
  public:
    // Returns false on error
    virtual bool OnPressed(int FKey /*in*/,
                           Text& Line /*in+out*/,
                           size_t& Cursor /*in+out*/,
                           EditorRange& R /*out*/) = 0;
    virtual ~FunKey();
  };
}

#endif // TEXTINPUT_COMPLETION_H
