#ifndef _WIN32

//===--- TerminalDisplayUnix.cpp - Output To UNIX Terminal ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for writing to a UNIX terminal. It tries to
//  support all "common" terminal types.
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#include "textinput/TerminalDisplayUnix.h"

#include <stdio.h>
// putenv not in cstdlib on Solaris
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <csignal>
#include <cstring>
#include <sstream>
#include <string>

#include "textinput/Color.h"
#include "textinput/Display.h"
#include "textinput/TerminalConfigUnix.h"

using std::signal;
using std::strstr;

namespace {
  textinput::TerminalDisplayUnix*& gTerminalDisplayUnix() {
    static textinput::TerminalDisplayUnix* S = 0;
    return S;
  }

  void InitRGB256(unsigned char rgb256[][3]) {
    // initialize the array with the expected standard colors:
    // (from http://frexx.de/xterm-256-notes)

    // this is not what I see, though it's supposedly the default:
    //   rgb[0][0] =   0; rgb[0][1] =   0; rgb[0][1] =   0;
    // use this instead, just to be on the safe side:
    rgb256[0][0] =  46; rgb256[0][1] =  52; rgb256[0][2] =  64;
    rgb256[1][0] = 205; rgb256[1][1] =   0; rgb256[1][2] =   0;
    rgb256[2][0] =   0; rgb256[2][1] = 205; rgb256[2][2] =   0;
    rgb256[3][0] = 205; rgb256[3][1] = 205; rgb256[3][2] =   0;
    rgb256[4][0] =   0; rgb256[4][1] =   0; rgb256[4][2] = 238;
    rgb256[5][0] = 205; rgb256[5][1] =   0; rgb256[5][2] = 205;
    rgb256[6][0] =   0; rgb256[6][1] = 205; rgb256[6][2] = 205;
    rgb256[7][0] = 229; rgb256[7][1] = 229; rgb256[7][2] = 229;

    // this is not what I see, though it's supposedly the default:
    //   rgb256[ 8][0] = 127; rgb256[ 8][1] = 127; rgb256[ 8][1] = 127;
    // use this instead, just to be on the safe side:
    rgb256[ 8][0] =   0; rgb256[ 8][1] =   0; rgb256[ 8][2] =   0;
    rgb256[ 9][0] = 255; rgb256[ 9][1] =   0; rgb256[ 9][2] =   0;
    rgb256[10][0] =   0; rgb256[10][1] = 255; rgb256[10][2] =   0;
    rgb256[11][0] = 255; rgb256[11][1] = 255; rgb256[11][2] =   0;
    rgb256[12][0] =  92; rgb256[12][1] =  92; rgb256[12][2] = 255;
    rgb256[13][0] = 255; rgb256[13][1] =   0; rgb256[13][2] = 255;
    rgb256[14][0] =   0; rgb256[14][1] = 255; rgb256[14][2] = 255;
    rgb256[15][0] = 255; rgb256[15][1] = 255; rgb256[15][2] = 255;

    // 6 intensity RGB
    static const int intensities[] = {0, 0x5f, 0x87, 0xaf, 0xd7, 0xff};
    int idx = 16;
    for (int r = 0; r < 6; ++r) {
      for (int g = 0; g < 6; ++g) {
        for (int b = 0; b < 6; ++b) {
          rgb256[idx][0] = intensities[r];
          rgb256[idx][1] = intensities[g];
          rgb256[idx][2] = intensities[b];
          ++idx;
        }
      }
    }

    // colors 232-255 are a grayscale ramp, intentionally leaving out
    // black and white
    for (unsigned char gray = 0; gray < 24; ++gray) {
      unsigned char level = (gray * 10) + 8;
      rgb256[232 + gray][0] = level;
      rgb256[232 + gray][1] = level;
      rgb256[232 + gray][2] = level;
    }
  }
} // unnamed namespace

extern "C" void TerminalDisplayUnix__handleResizeSignal(int) {
  gTerminalDisplayUnix()->HandleResizeSignal();
}

namespace textinput {
  // If input is not a tty don't write in tty-mode either.
  TerminalDisplayUnix::TerminalDisplayUnix():
    TerminalDisplay(TerminalConfigUnix::Get().IsInteractive()),
    fIsAttached(false), fNColors(16) {
    HandleResizeSignal();
    gTerminalDisplayUnix() = this;
    signal(SIGWINCH, TerminalDisplayUnix__handleResizeSignal);
#ifdef TCSANOW
    TerminalConfigUnix::Get().TIOS()->c_lflag &= ~(ECHO);
    TerminalConfigUnix::Get().TIOS()->c_lflag |= ECHOCTL|ECHOKE|ECHOE;
#endif
    const char* TERM = getenv("TERM");
    if (TERM &&  strstr(TERM, "256")) {
      fNColors = 256;
    }
  }

  TerminalDisplayUnix::~TerminalDisplayUnix() {
    Detach();
  }

  void
  TerminalDisplayUnix::HandleResizeSignal() {
#ifdef TIOCGWINSZ
    struct winsize sz;
    int ret = ioctl(fileno(stdout), TIOCGWINSZ, (char*)&sz);
    if (!ret && sz.ws_col) {
      SetWidth(sz.ws_col);

      // Export what we found.
      std::stringstream s;
      s << sz.ws_col;
      setenv("COLUMS", s.str().c_str(), 1 /*overwrite*/);
      s.clear();
      s << sz.ws_row;
      setenv("LINES", s.str().c_str(), 1 /*overwrite*/);
    }
#else
    // try $COLUMNS
    const char* COLUMNS = getenv("COLUMNS");
    if (COLUMNS) {
      long width = atol(COLUMNS);
      if (width > 4 && width < 1024*16) {
        SetWidth(width);
      }
    }
#endif
  }

  void
  TerminalDisplayUnix::SetColor(char CIdx, const Color& C) {
    if (!IsTTY()) return;

    // Default color, reset previous bold etc.
    static const char text[] = {(char)0x1b, '[', '0', 'm'};
    WriteRawString(text, sizeof(text));

    if (CIdx == 0) return;

    if (fNColors == 256) {
      int ANSIIdx = GetClosestColorIdx256(C);
      static const char preamble[] = {'\x1b', '[', '3', '8', ';', '5', ';', 0};
      std::string buf(preamble);
      if (ANSIIdx > 100) {
        buf += '0' + (ANSIIdx / 100);
      }
      if (ANSIIdx > 10) {
        buf += '0' + ((ANSIIdx / 10) % 10);
      }
      buf += '0' + (ANSIIdx % 10);
      buf +=  "m";
      WriteRawString(buf.c_str(), buf.length());
    } else {
      int ANSIIdx = GetClosestColorIdx16(C);
      char buf[] = {'\x1b', '[', '3', static_cast<char>('0' + (ANSIIdx % 8)), 'm', 0};
      if (ANSIIdx > 7) buf[2] += 6;
      WriteRawString(buf, 5);
    }

    if (C.fModifiers & Color::kModUnderline) {
      WriteRawString("\033[4m", 4);
    }
    if (C.fModifiers & Color::kModBold) {
      WriteRawString("\033[1m", 4);
    }
    if (C.fModifiers & Color::kModInverse) {
      WriteRawString("\033[7m", 4);
    }

  }

  void
  TerminalDisplayUnix::MoveFront() {
    static const char text[] = {(char)0x1b, '[', '1', 'G'};
    if (!IsTTY()) return;
    WriteRawString(text, sizeof(text));
  }

  void
  TerminalDisplayUnix::MoveInternal(char What, size_t n) {
    static const char cmd[] = "\x1b[";
    if (!IsTTY()) return;
    std::string text;
    for (size_t i = 0; i < n; ++i) {
      text += cmd;
      text += What;
    }
    WriteRawString(text.c_str(), text.length());
  }

  void
  TerminalDisplayUnix::MoveUp(size_t nLines /* = 1 */) {
    MoveInternal('A', nLines);
  }

  void
  TerminalDisplayUnix::MoveDown(size_t nLines /* = 1 */) {
    MoveInternal('B', nLines);
  }

  void
  TerminalDisplayUnix::MoveRight(size_t nCols /* = 1 */) {
    MoveInternal('C', nCols);
  }

  void
  TerminalDisplayUnix::MoveLeft(size_t nCols /* = 1 */) {
    MoveInternal('D', nCols);
  }

  void
  TerminalDisplayUnix::EraseToRight() {
    static const char text[] = {(char)0x1b, '[', 'K'};
    if (!IsTTY()) return;
    WriteRawString(text, sizeof(text));
  }

  void
  TerminalDisplayUnix::WriteRawString(const char *text, size_t len) {
    if (write(fileno(stdout), text, len) == -1) {
      // Silence Ubuntu's "unused result". We don't care if it fails.
    }
  }

  void
  TerminalDisplayUnix::ActOnEOL() {
    if (!IsTTY()) return;
    WriteRawString(" \b", 2);
    //MoveUp();
  }

  void
  TerminalDisplayUnix::Attach() {
    // set to noecho
    if (fIsAttached) return;
    fflush(stdout);
    TerminalConfigUnix::Get().Attach();
    fWritePos = Pos();
    fWriteLen = 0;
    fIsAttached = true;
  }

  void
  TerminalDisplayUnix::Detach() {
    if (!fIsAttached) return;
    fflush(stdout);
    TerminalConfigUnix::Get().Detach();
    TerminalDisplay::Detach();
    fIsAttached = false;
  }

  int
  TerminalDisplayUnix::GetClosestColorIdx16(const Color& C) {
    int r = C.fR;
    int g = C.fG;
    int b = C.fB;
    int sum = r + g + b;
    r = r > sum / 4;
    g = g > sum / 4;
    b = b > sum / 4;

    // ANSI:
    return r + (g * 2) + (b * 4);
    // ! ANSI:
    // return (r * 4) + (g * 2) + b;
  }

  int
  TerminalDisplayUnix::GetClosestColorIdx256(const Color& C) {
    static unsigned char rgb256[256][3] = {{0}};
    if (rgb256[0][0] == 0) {
      InitRGB256(rgb256);
    }

    // Find the closest index.
    // A: the closest color match (square of geometric distance in RGB)
    // B: the closest brightness match
    // Treat them equally, which suppresses differences
    // in color due to squared distance.

    // start with black:
    int idx = 0;
    unsigned int r = C.fR;
    unsigned int g = C.fG;
    unsigned int b = C.fB;
    unsigned int graylvl = (r + g + b)/3;
    long mindelta = (r*r + g*g + b*b) + graylvl;
    if (mindelta) {
      for (unsigned int i = 0; i < 256; ++i) {
        long delta = (rgb256[i][0] + rgb256[i][1] + rgb256[i][2])/3 - graylvl;
        if (delta < 0) delta = -delta;
        delta += (r-rgb256[i][0])*(r-rgb256[i][0]) +
        (g-rgb256[i][1])*(g-rgb256[i][1]) +
        (b-rgb256[i][2])*(b-rgb256[i][2]);

        if (delta < mindelta) {
          mindelta = delta;
          idx = i;
          if (mindelta == 0) break;
        }
      }
    }
    return idx;
  }

}

#endif // #ifndef _WIN32
