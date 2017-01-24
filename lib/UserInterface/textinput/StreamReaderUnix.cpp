//===--- TerminalReaderUnix.cpp - Input From UNIX Terminal ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface reading from a UNIX terminal. It tries to
//  support all common terminal types.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#ifndef _WIN32

#include "textinput/StreamReaderUnix.h"

#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>
#include <termios.h>
#include <stdio.h>
#ifdef __APPLE__
#include <errno.h> // For EINTR
#endif

#include <cctype>
#include <cstring>
#include <map>
#include <list>
#include <utility>

#include "textinput/InputData.h"
#include "textinput/KeyBinding.h"
#include "textinput/TerminalConfigUnix.h"
#include "textinput/TextInputContext.h"

namespace {
  using namespace textinput;
  using std::memset; // FD_ZERO

  class Rewind {
  public:
    Rewind(std::queue<char>& rab, InputData::EExtendedInput& ret):
    RAB(rab), Ret(ret) {}

    ~Rewind() {
      if (Ret != InputData::kEIUninitialized) return;
      // RAB.push(0x1b); already handled by ProcessCSI returning false.
      while (!Q.empty()) {
        RAB.push(Q.front());
        Q.pop();
      }
    }

    void push(char C) { Q.push(C); }

  private:
    std::queue<char> Q;
    std::queue<char>& RAB;
    InputData::EExtendedInput& Ret;
  };

  class ExtKeyMap {
  public:
    ExtKeyMap& operator[](char k) {
      std::map<char, ExtKeyMap*>::iterator I = Map.find(k);
      ExtKeyMap* N = 0;
      if (I == Map.end()) {
        N = &BumpAlloc();
        Map.insert(std::make_pair(k, N));
      } else {
        N = I->second;
      }
      return *N;
    }
    ExtKeyMap& operator=(InputData::EExtendedInput ei) {
      T.EI = ei; return *this; }
    void Set(InputData::EExtendedInput ei, char m = 0) {
      T.EI = ei; T.Mod = m; }

    bool empty() const { return Map.empty(); }
    bool haveExtInp() const { return empty(); } // no sub-tree
    InputData::EExtendedInput getExtInp() const { return T.EI; }
    char getMod() const { return T.Mod; }

    ExtKeyMap* find(char c) const {
      std::map<char, ExtKeyMap*>::const_iterator I = Map.find(c);
      if (I == Map.end()) return 0;
      return I->second;
    }

    class EKMHolder {
    public:
      EKMHolder(): Watermark(kChunkSize) {}
      ~EKMHolder() {
        for (std::list<ExtKeyMap*>::iterator i = Heap.begin(), e = Heap.end();
             i != e; ++i) {
          delete [] *i;
        }
      }
      ExtKeyMap& next() {
        if (Watermark == kChunkSize) {
          ExtKeyMap* N = new ExtKeyMap[kChunkSize]();
          Heap.push_back(N);
          Watermark = 0;
        }
        return Heap.back()[Watermark++];
      }
    private:
      enum EChunkSize { kChunkSize = 100 };
      std::list<ExtKeyMap*> Heap;
      size_t Watermark;
    };

    static ExtKeyMap& BumpAlloc() {
      static EKMHolder S;
      return S.next();
    }

  private:
    std::map<char, ExtKeyMap*> Map;
    struct T_ {
      T_(): EI(InputData::kEIUninitialized), Mod(0) {}
      InputData::EExtendedInput EI;
      char Mod;
    } T;
  };
} // unnamed namespace

namespace textinput {
  StreamReaderUnix::StreamReaderUnix():
    fHaveInputFocus(false), fIsTTY(isatty(fileno(stdin))) {
#ifdef TCSANOW
    // ~ISTRIP - do not strip 8th char bit
    // ~IXOFF - software flow ctrl disabled for input queue
    TerminalConfigUnix::Get().TIOS()->c_iflag &= ~(ISTRIP|IXOFF);
    // BRKINT - flush i/o and send SIGINT
    // INLCR - translate NL to CR
    TerminalConfigUnix::Get().TIOS()->c_iflag |= BRKINT | INLCR;
    // ~ICANON - non-canonical = input available immediately, no EOL needed, no processing, line editing disabled
    // ~ISIG - don't sent signals on input chars
    // ~TOSTOP - don't send SIGTTOU
    // ~IEXTEN - disable implementation-defined input processing, don't process spec chars (EOL2, LNEXT...)
    TerminalConfigUnix::Get().TIOS()->c_lflag &= ~(ICANON|ISIG|TOSTOP|IEXTEN);
    TerminalConfigUnix::Get().TIOS()->c_cc[VMIN] = 1; // minimum chars to read in non-canonical mode
    TerminalConfigUnix::Get().TIOS()->c_cc[VTIME] = 0; // waits indefinitely for VMIN chars (blocking)
#endif
  }

  StreamReaderUnix::~StreamReaderUnix() {
    ReleaseInputFocus();
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Attach to terminal, set the proper configuration.
  void
  StreamReaderUnix::GrabInputFocus() {
    // set to raw i.e. unbuffered
    if (fHaveInputFocus) return;
    TerminalConfigUnix::Get().Attach();
    fHaveInputFocus = true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Detach from terminal, set the old configuration.
  void
  StreamReaderUnix::ReleaseInputFocus() {
    // set to buffered
    if (!fHaveInputFocus) return;
    TerminalConfigUnix::Get().Detach();
    fHaveInputFocus = false;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Test or wait for available input
  ///
  /// \param[in] wait blocking wait on input
  ///
  /// Wait true - block, wait false - poll
  bool
  StreamReaderUnix::HavePendingInput(bool wait) {
    if (!fReadAheadBuffer.empty())
      return true;
    fd_set PollSet;
    FD_ZERO(&PollSet);
    FD_SET(fileno(stdin), &PollSet);
    timeval timeout = {0,0}; // sec, musec
    int avail = select(fileno(stdin) /*fd*/ + 1, &PollSet, 0, 0,
                       wait ? 0 : &timeout);
    return (avail == 1);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Process Control Sequence Introducer commands (equivalent to ESC [)
  ///
  /// \param[in] in input char / data
  bool
  StreamReaderUnix::ProcessCSI(InputData& in) {
    static ExtKeyMap gExtKeyMap;
    if (gExtKeyMap.empty()) {
      // Gnome xterm
      gExtKeyMap['[']['A'] = InputData::kEIUp;
      gExtKeyMap['[']['B'] = InputData::kEIDown;
      gExtKeyMap['[']['C'] = InputData::kEIRight;
      gExtKeyMap['[']['D'] = InputData::kEILeft;
      gExtKeyMap['[']['F'] = InputData::kEIEnd; // Savannah 83478
      gExtKeyMap['[']['H'] = InputData::kEIHome;
      gExtKeyMap['[']['O']['F'] = InputData::kEIEnd;
      gExtKeyMap['[']['O']['H'] = InputData::kEIHome;
      gExtKeyMap['[']['1']['~'] = InputData::kEIHome;
      gExtKeyMap['[']['2']['~'] = InputData::kEIIns;
      gExtKeyMap['[']['3']['~'] = InputData::kEIDel;
      gExtKeyMap['[']['4']['~'] = InputData::kEIEnd;
      gExtKeyMap['[']['5']['~'] = InputData::kEIPgUp;
      gExtKeyMap['[']['6']['~'] = InputData::kEIPgDown;
      gExtKeyMap['[']['1'][';']['5']['A'].Set(InputData::kEIUp,
                                         InputData::kModCtrl);
      gExtKeyMap['[']['1'][';']['5']['B'].Set(InputData::kEIDown,
                                         InputData::kModCtrl);
      gExtKeyMap['[']['1'][';']['5']['C'].Set(InputData::kEIRight,
                                         InputData::kModCtrl);
      gExtKeyMap['[']['1'][';']['5']['D'].Set(InputData::kEILeft,
                                         InputData::kModCtrl);

      // MacOS
      gExtKeyMap['O']['A'] = InputData::kEIUp;
      gExtKeyMap['O']['B'] = InputData::kEIDown;
      gExtKeyMap['O']['C'] = InputData::kEIRight;
      gExtKeyMap['O']['D'] = InputData::kEILeft;
      gExtKeyMap['O']['F'] = InputData::kEIEnd;
      gExtKeyMap['O']['H'] = InputData::kEIHome;
      gExtKeyMap['[']['5']['C'].Set(InputData::kEIRight, InputData::kModCtrl);
      gExtKeyMap['[']['5']['D'].Set(InputData::kEILeft, InputData::kModCtrl);
    }

    InputData::EExtendedInput ret = InputData::kEIUninitialized;
    char mod = 0;
    Rewind rwd(fReadAheadBuffer, ret);
    ExtKeyMap* EKM = &gExtKeyMap;
    while (EKM) {
      if (EKM->haveExtInp()) {
        ret = EKM->getExtInp();
        mod = EKM->getMod();
        EKM = 0;
      } else {
        char c1 = ReadRawCharacter();
        rwd.push(c1);
        EKM = EKM->find(c1);
      }
    }
    in.SetExtended(ret);
    in.SetModifier(mod);
    return ret != InputData::kEIUninitialized;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Read one char from stdin. Converts the read char to InputData
  ///
  /// \param[in] nRead number of already read characters. Increment after reading
  /// \param[in] in input char / data to be filled out
  bool
  StreamReaderUnix::ReadInput(size_t& nRead, InputData& in) {
    int c = ReadRawCharacter();
    in.SetModifier(InputData::kModNone);
    if (c == -1) { // non-character value, EOF negative
      in.SetExtended(InputData::kEIEOF);
    } else if (c == 0x1b) { // ESC
      // Only try to process CSI if Esc does not have a meaning by itself.
      // by itself.
      if (GetContext()->GetKeyBinding()->IsEscCommandEnabled()
          || !ProcessCSI(in)) {
        in.SetExtended(InputData::kEIEsc);
      }
    } else if (isprint(c)) { // c >= 0x20(32) && c < 0x7f(127)
      in.SetRaw(c);
    } else if (c < 32 || c == (char)127 /* ^?, DEL on MacOS */) { // non-printable
      if (c == 13) { // 0x0d CR (INLCR - NL converted to CR)
        in.SetExtended(InputData::kEIEnter);
      } else { // mark CTRL pressed if other non-print char
        in.SetRaw(c);
        in.SetModifier(InputData::kModCtrl);
      }
    } else {
      // woohoo, what's that?!
      in.SetRaw(c);
    }
    ++nRead;
    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Read one character from stdin. Block if not available.
  int
  StreamReaderUnix::ReadRawCharacter() {
    char buf;
    if (!fReadAheadBuffer.empty()) {
      buf = fReadAheadBuffer.front();
      fReadAheadBuffer.pop();
    } else {
      ssize_t ret = read(fileno(stdin), &buf, 1);
#ifdef __APPLE__
      // Allow a debugger to be attached and used on OS X.
      while (ret == -1 && errno == EINTR)
        ret = read(fileno(stdin), &buf, 1);
#endif
      if (ret != 1) return -1;
    }
    return buf;
  }
}

#endif // ifndef _WIN32
