//===--- TextInputContext.cpp - Object Holder -------------------*- C++ -*-===//
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

#include "textinput/TextInputContext.h"
#include "textinput/Editor.h"
#include "textinput/KeyBinding.h"
#include "textinput/SignalHandler.h"
#include "textinput/Reader.h"
#include "textinput/Display.h"
#include "textinput/History.h"
#include "textinput/Color.h"

textinput::TextInputContext::TextInputContext(TextInput* ti,
                                              const char* histFile):
fTextInput(ti), fBind(nullptr), fEdit(nullptr), fSignal(nullptr), fColor(nullptr), fHist(nullptr),
fTabCompletion(nullptr), fFunKey(nullptr), fCursor(0) {
  fHist = new History(histFile);
  fEdit = new Editor(this);
  fBind = new KeyBinding();
  fSignal = new SignalHandler();
}

textinput::TextInputContext::~TextInputContext() {
  delete fBind;
  delete fEdit;
  delete fSignal;
  delete fHist;
}

textinput::TextInputContext&
textinput::TextInputContext::AddReader(Reader& R) {
  fReaders.push_back(&R);
  R.SetContext(this);
  return *this;
}

textinput::TextInputContext&
textinput::TextInputContext::AddDisplay(Display& D) {
  fDisplays.push_back(&D);
  D.SetContext(this);
  return *this;
}

// vtable goes here.
textinput::Colorizer::~Colorizer() {}

textinput::Display::~Display() {
  Detach();
}

textinput::Reader::~Reader() {
  ReleaseInputFocus();
}
