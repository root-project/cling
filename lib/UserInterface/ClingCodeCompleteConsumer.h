#ifndef CLINGCODECOMPLETECONSUMER_H
#define CLINGCODECOMPLETECONSUMER_H

#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/CodeCompleteOptions.h"

#include "textinput/Text.h"
#include "textinput/Editor.h"
#include "textinput/Callbacks.h"

using namespace clang;
using namespace textinput;

class ClingCodeCompleteConsumer : public clang::CodeCompleteConsumer {
  clang::CodeCompletionTUInfo CCTUInfo;
public:
  ClingCodeCompleteConsumer(const CodeCompleteOptions &Opts)
    : CodeCompleteConsumer(Opts, false), CCTUInfo(new GlobalCodeCompletionAllocator) { }
  ~ClingCodeCompleteConsumer() {}
  CodeCompletionAllocator &getAllocator() override { return CCTUInfo.getAllocator();}
  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }
};

class ClingTabCompletion : public textinput::TabCompletion {
  ClingCodeCompleteConsumer* fCodeCompleteConsumer;
public:
  void SetConsumer(ClingCodeCompleteConsumer* CCCC) {
    fCodeCompleteConsumer = CCCC;
  }
  bool Complete(Text& Line /*in+out*/,
                          size_t& Cursor /*in+out*/,
                          EditorRange& R /*out*/,
                          std::vector<std::string>& DisplayCompletions /*out*/) override {
    return true;
  }
};

#endif
