//===--- History.cpp - Previously Entered Lines -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for setting and retrieving previously
//  entered input, with a persistent backend (i.e. a history file).
//
//  Axel Naumann <axel@cern.ch>, 2011-05-12
//===----------------------------------------------------------------------===//

#include "textinput/History.h"
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef WIN32
# include <stdio.h>
extern "C" unsigned long __stdcall GetCurrentProcessId(void);
#else
# include <unistd.h>
#endif

namespace textinput {
  History::History(const char* filename):
    fHistFileName(filename ? filename : ""), fMaxDepth((size_t) -1),
    fPruneLength(0), fNumHistFileLines(0) {
    // Create a history object, initialize from filename if the file
    // exists. Append new lines to filename taking into account the
    // maximal number of lines allowed by SetMaxDepth().
    if (filename) ReadFile(filename);
  }

  History::~History() {}

  void
  History::AddLine(const std::string& line) {
    // Add a line to entries and file.
    if (line.empty()) return;
    fEntries.push_back(line);
    AppendToFile();
  }

  void
  History::ReadFile(const char* FileName) {
    // Inject all lines of FileName.
    // Intentionally ignore fMaxDepth
    std::ifstream InHistFile(FileName);
    if (!InHistFile) return;
    std::string line;
    while (std::getline(InHistFile, line)) {
      while (!line.empty()) {
        size_t len = line.length();
        char c = line[len - 1];
        if (c != '\n' && c != '\r') break;
        line.erase(len - 1);
      }
      if (!line.empty()) {
        fEntries.push_back(line);
      }
    }
    fNumHistFileLines = fEntries.size();
  }

  void
  History::AppendToFile() {
    // Write last entry to hist file.
    // Prune if needed.
    if (fHistFileName.empty() || !fMaxDepth) return;

    // Calculate prune length to use
    size_t nPrune = fPruneLength;
    if (nPrune == (size_t)kPruneLengthDefault) {
      nPrune = (size_t)(fMaxDepth * 0.8);
    } else if (nPrune > fMaxDepth) {
      nPrune = fMaxDepth - 1; // fMaxDepth is guaranteed to be > 0.
    }

    // Don't check for the actual line count of the history file after every
    // single line. Once every 50% on the way between nPrune and fMaxDepth is
    // enough.
    if (fNumHistFileLines < fMaxDepth
        && (fNumHistFileLines % (fMaxDepth - nPrune)) == 0) {
      fNumHistFileLines = 0;
      std::string line;
      std::ifstream in(fHistFileName.c_str());
      while (std::getline(in, line))
        ++fNumHistFileLines;
    }

    size_t numLines = fNumHistFileLines;
    if (numLines >= fMaxDepth) {
      // Prune! But don't simply write our lines - other processes might have
      // added their own.
      std::string line;
      std::ifstream in(fHistFileName.c_str());
      std::stringstream pruneFileNameStream;
      pruneFileNameStream << fHistFileName + "_prune"
#if _WIN32
                          << ::GetCurrentProcessId();
#else
                          << ::getpid();
#endif
      std::ofstream out(pruneFileNameStream.str().c_str());
      if (out) {
        if (in) {
          while (numLines >= nPrune && std::getline(in, line)) {
            // skip
            --numLines;
          }
          while (std::getline(in, line)) {
            out << line << '\n';
          }
        }
        out << fEntries.back() << '\n';
        in.close();
        out.close();
#ifdef WIN32
        ::_unlink(fHistFileName.c_str());
#else
        ::unlink(fHistFileName.c_str());
#endif
        if (::rename(pruneFileNameStream.str().c_str(), fHistFileName.c_str()) == -1){
           std::cerr << "ERROR in textinput::History::AppendToFile(): "
              "cannot rename " << pruneFileNameStream.str() << " to " << fHistFileName;
        }
        fNumHistFileLines = nPrune;
      }
    } else {
      std::ofstream out(fHistFileName.c_str(), std::ios_base::app);
      out << fEntries.back() << '\n';
      ++fNumHistFileLines;
    }
  }
}
