#ifndef _WIN32

//===--- TerminalConfigUnix.cpp - termios storage -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TerminalReader and TerminalDisplay need to reset the terminal configuration
// upon destruction, to leave the terminal as before. To avoid a possible
// misunderstanding of what "before" means, centralize their storage of the
// previous termios and have them share it.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#include "textinput/TerminalConfigUnix.h"

#include <termios.h>
#include <unistd.h>
#include <stdio.h>
#include <cstring>

using namespace textinput;
using std::memcpy;
using std::signal;
using std::raise;

namespace {
void
TerminalConfigUnix__handleSignal(int signum) {
  // Clean up before we are killed.
  TerminalConfigUnix::Get().HandleSignal(signum);
}
}

const int TerminalConfigUnix::fgSignals[kNumHandledSignals] = {
  // SIGKILL: can't handle by definition
  // Order: most to least common signal
  SIGTERM,
  SIGABRT,
  SIGSEGV,
  SIGILL,
  SIGBUS
};

TerminalConfigUnix&
TerminalConfigUnix::Get() {
  static TerminalConfigUnix s;
  return s;
}

TerminalConfigUnix::TerminalConfigUnix():
  fIsAttached(false), fFD(fileno(stdin)), fOldTIOS(), fConfTIOS() {
#ifdef TCSANOW
  fOldTIOS = new termios;
  fConfTIOS = new termios;
  tcgetattr(fFD, fOldTIOS);
  *fConfTIOS = *fOldTIOS;
#endif
  for (int i = 0; i < kNumHandledSignals; ++i) {
    fPrevHandler[i] = signal(fgSignals[i], TerminalConfigUnix__handleSignal);
  }
}

TerminalConfigUnix::~TerminalConfigUnix() {
  // Restore signals and detach.
  for (int i = 0; i < kNumHandledSignals; ++i) {
    if (fPrevHandler[i]) {
      signal(fgSignals[i], fPrevHandler[i]);
    } else {
      // default:
      signal(fgSignals[i], SIG_DFL);
    }
  }
  Detach();
  delete fOldTIOS;
  delete fConfTIOS;
}

void
TerminalConfigUnix::HandleSignal(int signum) {
  // Process has received a fatal signal, detach.
  bool sSignalHandlerActive = false;
  if (!sSignalHandlerActive) {
    sSignalHandlerActive = true;
    Detach();
    // find previous signal handler index:
    for (int i = 0; i < kNumHandledSignals; ++i) {
      if (fgSignals[i] == signum) {
        // Call previous signal handler if it exists.
        if (fPrevHandler[i]) {
          fPrevHandler[i](signum);
          // should not end up here...
          sSignalHandlerActive = false;
          return;
        } else break;
      }
    }
  }

  // No previous handler found, re-raise to get default handling:
  signal(signum, SIG_DFL); // unregister ourselves
  raise(signum); // terminate through default handler

  // No need to recover our state; there will be no "next time":
  // the signal raised above will cause the program to quit.
  //signal(signum, TerminalConfigUnix__handleSignal);
  //sSignalHandlerActive = false;
}

void
TerminalConfigUnix::Attach() {
  if (fIsAttached) return;
#ifdef TCSANOW
  if (IsInteractive()) {
    tcsetattr(fFD, TCSANOW, fConfTIOS);
  }
#endif
  fIsAttached = true;
}

void
TerminalConfigUnix::Detach() {
  // Reset the terminal configuration.
  if (!fIsAttached) return;
#ifdef TCSANOW
  if (IsInteractive()) {
    tcsetattr(fFD, TCSANOW, fOldTIOS);
  }
#endif
  fIsAttached = false;
}

bool TerminalConfigUnix::IsInteractive() const {
  // Whether both stdin and stdout are attached to a tty.
  return isatty(fileno(stdin)) && isatty(fileno(stdout))
    && (getpgrp() == tcgetpgrp(STDOUT_FILENO));
}



#endif // ifndef _WIN32
