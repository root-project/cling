
//===--- TerminalConfigUnix.cpp - termios storage ---------------*- C++ -*-===//
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

#ifndef TEXTINPUT_UNIXTERMINALSETTINGS_H
#define TEXTINPUT_UNIXTERMINALSETTINGS_H

#include <csignal>

struct termios;

namespace textinput {
#if defined(__sun) && defined(__SVR4) && !defined(sig_t)
// Solaris doesn't define sig_t
typedef SIG_TYP sig_t;
#endif
#if defined(_AIX) && !defined(sig_t)
// AIX doesn't define sig_t
typedef void (*sig_t)(int);
#endif

class TerminalConfigUnix {
private:
   TerminalConfigUnix(const TerminalConfigUnix&); // not implemented
   TerminalConfigUnix& operator=(const TerminalConfigUnix&); // not implemented
public:
  ~TerminalConfigUnix();

  static TerminalConfigUnix& Get();
  termios* TIOS() { return fConfTIOS; }

  void Attach();
  void Detach();
  bool IsAttached() const { return fIsAttached; }
  bool IsInteractive() const;

  void HandleSignal(int signum);

private:
  TerminalConfigUnix();

  bool fIsAttached; // whether fConfTIOS is active.
  int fFD; // file descriptor
  enum { kNumHandledSignals = 5 }; // number of fPrevHandler entries
  static const int
    fgSignals[kNumHandledSignals]; // signal nums, order as in fPrevHandler
  sig_t fPrevHandler[kNumHandledSignals]; // next signal handler to call
  termios* fOldTIOS; // tty configuration before grabbing
  termios* fConfTIOS; // tty configuration while active
};

}
#endif // TEXTINPUT_UNIXTERMINALSETTINGS_H
