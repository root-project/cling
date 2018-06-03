//===--- SignalHandler.h - Signal Emission ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for emitting signals.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef TEXTINPUT_SIGNALHANDLER_H
#define TEXTINPUT_SIGNALHANDLER_H

namespace textinput {
  // Signalling interface.
  class SignalHandler {
  public:
    SignalHandler() {}
    ~SignalHandler() {}

    void EmitCtrlZ();
  };
}

#endif // TEXTINPUT_SIGNALHANDLER_H
