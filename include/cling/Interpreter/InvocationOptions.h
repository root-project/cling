//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_INVOCATIONOPTIONS_H
#define CLING_INVOCATIONOPTIONS_H

#include <string>
#include <vector>

namespace cling {
  class InvocationOptions {
  public:
    InvocationOptions():
      ErrorOut(false), NoLogo(false), ShowVersion(false), Verbose(false),
      Help(false), MetaString(".") {}
    bool ErrorOut;
    bool NoLogo;
    bool ShowVersion;
    bool Verbose;
    bool Help;

    /// \brief A line starting with this string is assumed to contain a
    ///        directive for the MetaProcessor. Defaults to "."
    std::string MetaString;

    std::vector<std::string> LibsToLoad;
    std::vector<std::string> LibSearchPath;
    std::vector<std::string> Inputs;

    static InvocationOptions CreateFromArgs(int argc, const char* const argv[],
                                            std::vector<unsigned>& leftoverArgs
                                            /* , Diagnostic &Diags */);

    void PrintHelp();
  };
}

#endif // INVOCATIONOPTIONS
