//===--- examplemain.cpp - Example for textinput use ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file illustrates the use of the textinput library.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#include "textinput/TextInput.h"
#include "textinput/StreamReader.h"
#include "textinput/TerminalDisplay.h"
#include "textinput/Color.h"
#include "textinput/Text.h"

#ifdef WIN32
#include "Windows.h"
#endif

extern "C" int printf(const char*,...);

using namespace textinput;

class MyCol: public Colorizer {
public:
  void ProcessTextChange(EditorRange& Modification,
               Text& input) {
    size_t start = Modification.fEdit.fStart;
    if (start > 6) start -= 5; // not -= 6: we require at least one changed char
    else start = 0;
    size_t P = input.GetText().find("return", start, 6);
    if (P != std::string::npos) {
      for (size_t i = P; i < P + 6; ++i) {
        input.SetColor(i, 1);
      }
      Modification.fDisplay.Extend(Range(P, 6));
    }
  }

  void ProcessPromptChange(Text& prompt) {
    prompt.SetColor(Range::AllText(), 1);
  }

  void ProcessCursorChange(size_t Cursor, Text& input) {};
  bool GetColor(char C, Color& Col) {
    if (C == 1) Col = Color(255,0,0,0);
    else Col = Color(127,127,127,0);
    return true;
  }
};

int main (int argc, const char * argv[]) {
  using namespace textinput;
  StreamReader* R = StreamReader::Create();
  TerminalDisplay* D = TerminalDisplay::Create();
  TextInput TI(*R, *D, "textinput_history");
  TI.SetMaxPendingCharsToRead(10);
  TI.SetPrompt("Hello$ ");
  TI.SetColorizer(new MyCol);
  std::string line;
  do {
    while (!TI.AtEOL()) {
      while (!TI.AtEOL() && TI.HavePendingInput()) {
        TI.ReadInput();
      }
#ifdef WIN32
      Sleep(10);
#else
      usleep(100);
#endif
    }
    TI.TakeInput(line);
    printf("INPUT: --BEGIN--%s--END--\n", line.c_str());
    if (line == "h") TI.HideInput(!TI.IsInputHidden());
  } while (!TI.AtEOF() && line != ".q");

  delete R;
  delete D;
  return 0;
}
