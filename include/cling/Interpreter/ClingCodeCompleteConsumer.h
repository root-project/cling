#ifndef CLINGCODECOMPLETECONSUMER_H
#define CLINGCODECOMPLETECONSUMER_H

#include "clang/Sema/CodeCompleteConsumer.h"

using namespace clang;

class ClingCodeCompleteConsumer : public CodeCompleteConsumer {
  /// \brief The raw output stream.
  raw_ostream &OS;
  CodeCompletionTUInfo CCTUInfo;

public:
  /// \brief Create a new printing code-completion consumer that prints its
  /// results to the given raw output stream.
  ClingCodeCompleteConsumer(const CodeCompleteOptions &CodeCompleteOpts,
                               raw_ostream &OS)
    : CodeCompleteConsumer(CodeCompleteOpts, false), OS(OS),
      CCTUInfo(new GlobalCodeCompletionAllocator) {}

  ~ClingCodeCompleteConsumer() {}    

  /// \brief Prints the finalized code-completion results.
  void ProcessCodeCompleteResults(Sema &S, CodeCompletionContext Context,
                                  CodeCompletionResult *Results,
                                  unsigned NumResults) override;

  void ProcessOverloadCandidates(Sema &S, unsigned CurrentArg,
                                 OverloadCandidate *Candidates,
                                 unsigned NumCandidates) override;

  bool isResultFilteredOut(StringRef Filter, CodeCompletionResult Results) override;

  CodeCompletionAllocator &getAllocator() override {
    return CCTUInfo.getAllocator();
  }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }
};

#endif