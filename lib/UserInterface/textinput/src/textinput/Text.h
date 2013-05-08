//===--- Text.h - Colored Text ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for a string plus its characters' color
// indexes.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_TEXT_H
#define TEXTINPUT_TEXT_H
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include "textinput/Range.h"

namespace textinput {
  class Colorizer;
  using std::strlen;

  // A colored string.
  class Text {
  public:
    Text() {}
    Text(const char* S): fString(S), fColor(strlen(S)) {}
    Text(const std::string& S, char C = 0): fString(S), fColor(S.length(), C) {}

    const std::string& GetText() const { return fString; }
    const std::vector<char>& GetColors() const { return fColor; }
    std::vector<char>& GetColors() { return fColor; }
    char GetColor(size_t i) const { return fColor[i]; }
    size_t length() const { return fString.length(); }
    bool empty() const { return fColor.empty(); }

    void insert(size_t pos, char C) {
      // Insert C at pos, set to default color.
      fString.insert(pos, 1, C); fColor.insert(fColor.begin() + pos, 0);
    }
    void insert(size_t pos, const std::string& S) {
      // Inset S at pos, set to default color.
      size_t len = S.length();
      fColor.insert(fColor.begin() + pos, len, 0);
      fString.insert(pos, S);
    }
    void erase(size_t pos, size_t len = 1) {
      // Erase len characters starting at pos.
      fString.erase(pos, len);
      fColor.erase(fColor.begin() + pos, fColor.begin() + pos + len);
    }
    void clear() { fString.clear(); fColor.clear(); }

    void
    SetColor(const Range &R, char C) {
      // Set colors of characters in range R to C.
      size_t len = R.fLength;
      if (len == (size_t) -1) {
        len = length() - R.fStart;
      }
      std::fill_n(fColor.begin() + R.fStart, len, C);
    }

    char operator[](size_t i) const { return fString[i]; }
    char& operator[](size_t i) { return fString[i]; }

    Text& operator+=(char C) { insert(length(), C); return *this; }
    Text& operator=(const std::string& S) {
      // Assing string S to this, initialize with default colors.
      fColor.clear();
      fColor.resize(S.length());
      fString = S;
      return *this;
    }
  private:
    std::string fString; // actual text
    std::vector<char> fColor; // color index of chars; Colorizer converts to RGB
  };
}
#endif // TEXTINPUT_COLOR_H
