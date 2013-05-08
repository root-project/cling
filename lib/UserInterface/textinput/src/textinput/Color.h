//===--- Color.h - Color and Text Attributes --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the color / text attribute structure and an interface
//  that can set it.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_COLOR_H
#define TEXTINPUT_COLOR_H

#include "textinput/Range.h"

namespace textinput {
  class Editor;
  class Text;

  // Represents a color and display formatting for text.
  class Color {
  public:
    typedef unsigned char Intensity_t;

    // Bit flags for display modification.
    // Not necessarily supported by all displays.
    enum EModifiers {
      kModNone = 0,
      kModUnderline = 1,
      kModBold = 2,
      kModInverse = 4
    };

    Color(Intensity_t r = 127, Intensity_t g = 127, Intensity_t b = 127,
          char mod = 0): fR(r), fG(g), fB(b), fModifiers(mod) {}

    bool operator==(const Color& O) const {
      // Equality test.
      return fR == O.fR && fG == O.fG && fB == O.fB
        && fModifiers == O.fModifiers; }
    bool operator!=(const Color& O) const {
      // Inequality test.
      return fR != O.fR || fG != O.fG || fB != O.fB
        || fModifiers != O.fModifiers; }

    Intensity_t fR, fG, fB; // Intensity of the channels
    char fModifiers; // Character display modifiers; bitset of EModifiers
  };

  // Abtract interface for setting the color of text.
  class Colorizer {
  public:
    Colorizer() {}
    virtual ~Colorizer();
    virtual void ProcessTextChange(EditorRange& Modification, Text& input) = 0;
    virtual void ProcessPromptChange(Text& prompt) = 0;
    virtual void ProcessCursorChange(size_t /*Cursor*/, Text& /*input*/,
                                     Range& /*DisplayR*/) {}
    virtual bool GetColor(char C, Color& Col) = 0; // C == 0 is default color
    virtual char GetInfoColor() const { return 0; }
  };
}
#endif // TEXTINPUT_COLOR_H
