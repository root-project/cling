//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLINGCODECOMPLETECONSUMER_H
#define CLINGCODECOMPLETECONSUMER_H

#include "clang/Sema/CodeCompleteConsumer.h"

namespace cling {
  class Interpreter;
}

namespace clang {
  class PrintingCodeCompleteConsumer;
  class CodeCompletionTUInfo;
  class CodeCompletionAllocator;
  class DiagnosticConsumer;
}

namespace cling {
  class ClingCodeCompleteConsumer : public clang::PrintingCodeCompleteConsumer {
    clang::CodeCompletionTUInfo CCTUInfo;

  public:
    ClingCodeCompleteConsumer(const cling::Interpreter& Parent);
    ~ClingCodeCompleteConsumer();

    clang::CodeCompletionAllocator &getAllocator() override {
      return CCTUInfo.getAllocator();
    }

    clang::CodeCompletionTUInfo &getCodeCompletionTUInfo() override {
      return CCTUInfo;
    }

    void ProcessCodeCompleteResults(clang::Sema &S,
                                    clang::CodeCompletionContext Context,
                                    clang::CodeCompletionResult *Results,
                                    unsigned NumResults);
  };
}

#endif
