//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Output.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_os_ostream.h"

#include <iostream>

namespace cling {
  namespace utils {

    namespace {
      class ColoredOutput : public llvm::raw_os_ostream {
        bool m_Colorize = true;

        raw_ostream& changeColor(enum Colors colors, bool bold, bool bg) override {
          if (m_Colorize) {
            if (llvm::sys::Process::ColorNeedsFlush()) flush();
            if (const char* colorcode =
                    (colors == SAVEDCOLOR)
                        ? llvm::sys::Process::OutputBold(bg)
                        : llvm::sys::Process::OutputColor(colors, bold, bg))
              write(colorcode, strlen(colorcode));
          }
          return *this;
        }
        raw_ostream& resetColor() override {
          if (m_Colorize) {
            if (llvm::sys::Process::ColorNeedsFlush()) flush();
            if (const char* colorcode = llvm::sys::Process::ResetColor())
              write(colorcode, ::strlen(colorcode));
          }
          return *this;
        }

        raw_ostream& reverseColor() override {
          if (m_Colorize) {
            if (llvm::sys::Process::ColorNeedsFlush()) flush();

            if (const char* colorcode = llvm::sys::Process::OutputReverse())
              write(colorcode, ::strlen(colorcode));
          }
          return *this;
        }
        bool has_colors() const override { return m_Colorize; }
        bool is_displayed() const override { return m_Colorize; }
      public:

        ColoredOutput(std::ostream& Out, bool Unbufferd = true)
            : raw_os_ostream(Out) {
          if (Unbufferd) SetUnbuffered();
        }

        bool Colors(bool C) { m_Colorize = C; return m_Colorize; }
      };
    } // anonymous namespace

    llvm::raw_ostream& outs() {
      static ColoredOutput sOut(std::cout);
      return sOut;
    }

    llvm::raw_ostream& errs() {
      static ColoredOutput sErr(std::cerr);
      return sErr;
    }

    llvm::raw_ostream& log() {
      return cling::errs();
    }

    bool ColorizeOutput(unsigned Which) {
#define COLOR_FLAG(Fv, Fn) (Which == 8 ? llvm::sys::Process::Fn() : Which & Fv)
      bool colorStdout = COLOR_FLAG(1, StandardOutIsDisplayed);
      bool colorStderr = COLOR_FLAG(2, StandardErrIsDisplayed);
      // The following calls have side effects because they set m_Colorize!
      static_cast<ColoredOutput &>(outs()).Colors(colorStdout);
      static_cast<ColoredOutput &>(errs()).Colors(colorStderr);
      return colorStdout || colorStderr;
    }
  }
}
