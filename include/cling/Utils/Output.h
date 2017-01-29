//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_OUTPUT_H
#define CLING_OUTPUT_H

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace cling {
  namespace utils {
    ///\brief setup colorization of the output streams below
    ///
    ///\param[in] Which - Ored value of streams to colorize: 0 means none.
    /// 0 = Don't colorize any output
    /// 1 = Colorize cling::outs
    /// 2 = Colorize cling::errs
    /// 4 = Colorize cling::log (currently always the same as cling::errs)
    /// 8 = Colorize based on whether stdout/err are dsiplayed on tty or not.
    ///
    ///\returns Whether any output stream was colorized.
    ///
    bool ColorizeOutput(unsigned Which = 8);
    
    ///\brief The 'stdout' stream. llvm::raw_ostream wrapper of std::cout
    ///
    llvm::raw_ostream& outs();

    ///\brief The 'stderr' stream. llvm::raw_ostream wrapper of std::cerr
    ///
    llvm::raw_ostream& errs();

    ///\brief The 'logging' stream. Currently returns cling::errs().
    /// This matches clang & gcc prinitng to stderr for certain information.
    /// If the host process needs to keep stderr for itself or actual errors,
    /// then the function can be edited to return a separate stream.
    ///
    llvm::raw_ostream& log();

    ///\brief Wrappers around buffered llvm::raw_ostreams.
    /// outstring<N> with N > 0 are the fastest, using a stack allocated buffer.
    /// outstring<0> outputs directly into a std:string.

    template <size_t N = 512>
    class outstring {
      llvm::SmallString<N> m_Buf;
      llvm::raw_svector_ostream m_Strm;
    public:
      outstring() : m_Strm(m_Buf) {}
      template <typename T> llvm::raw_ostream& operator << (const T& V) {
        m_Strm << V;
        return m_Strm;
      }
      llvm::StringRef str() { return m_Strm.str(); }
      operator llvm::raw_ostream& () { return m_Strm; }
    };

    template <>
    class outstring<0> {
      std::string m_Str;
      llvm::raw_string_ostream m_Strm;
    public:
      outstring() : m_Strm(m_Str) {}
      template <typename T> llvm::raw_ostream& operator << (const T& V) {
        m_Strm << V;
        return m_Strm;
      }
      std::string& str() { return m_Strm.str(); }
      operator llvm::raw_ostream& () { return m_Strm; }
    };

    typedef outstring<512>  ostrstream;
    typedef outstring<128>  smallstream;
    typedef outstring<1024> largestream;
    typedef outstring<0>    stdstrstream;
  }
  using utils::outs;
  using utils::errs;
  using utils::log;

  using utils::ostrstream;
  using utils::smallstream;
  using utils::largestream;
  using utils::stdstrstream;
}

#endif // CLING_OUTPUT_H
