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
    TerminalConfigUnix::Get().TIOS()->c_iflag &= ~(ISTRIP|IXOFF);
    TerminalConfigUnix::Get().TIOS()->c_iflag |= BRKINT | INLCR;
    TerminalConfigUnix::Get().TIOS()->c_lflag &= ~(ICANON|ISIG|TOSTOP|IEXTEN);
    TerminalConfigUnix::Get().TIOS()->c_cc[VMIN] = 1;
    TerminalConfigUnix::Get().TIOS()->c_cc[VTIME] = 0;
#endif
  }

  StreamReaderUnix::~StreamReaderUnix() {
    ReleaseInputFocus();
  }

  void
  StreamReaderUnix::GrabInputFocus() {
    // set to raw i.e. unbuffered
    if (fHaveInputFocus) return;
    TerminalConfigUnix::Get().Attach();
    fHaveInputFocus = true;
  }

  void
  StreamReaderUnix::ReleaseInputFocus() {
    // set to buffered
    if (!fHaveInputFocus) return;
    TerminalConfigUnix::Get().Detach();
    fHaveInputFocus = false;
  }

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

  bool
  StreamReaderUnix::ReadInput(size_t& nRead, InputData& in) {
    int c = ReadRawCharacter();
    in.SetModifier(InputData::kModNone);
    if (c == -1) {
      in.SetExtended(InputData::kEIEOF);
    } else if (c == 0x1b) {
      // Only try to process CSI if Esc does not have a meaning
      // by itself.
      if (GetContext()->GetKeyBinding()->IsEscCommandEnabled()
          || !ProcessCSI(in)) {
        in.SetExtended(InputData::kEIEsc);
      }
    } else if (isprint(c)) {
      in.SetRaw(c);
    } else if (c < 32 || c == (char)127 /* ^?, DEL on MacOS */) {
      if (c == 13) {
        in.SetExtended(InputData::kEIEnter);
      } else {
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
  int
  StreamReaderUnix::ReadRawCharacter() {
    char buf;
    if (!fReadAheadBuffer.empty()) {
      buf = fReadAheadBuffer.front();
      fReadAheadBuffer.pop();
    } else {
       size_t ret = read(fileno(stdin), &buf, 1);
      if (ret != 1) return -1;
    }
    return buf;
  }
}

#endif // ifndef _WIN32
