#ifndef CLINGCODECOMPLETECONSUMER_H
#define CLINGCODECOMPLETECONSUMER_H

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/CodeCompleteOptions.h"
#include "clang/Sema/Sema.h"

#include "cling/Interpreter/Interpreter.h"

#include "textinput/Text.h"
#include "textinput/Editor.h"
#include "textinput/Callbacks.h"

using namespace clang;
using namespace textinput;
using namespace cling;

namespace textinput{
class ClingTabCompletion : public textinput::TabCompletion,
                           public clang::CodeCompleteConsumer {
  clang::CodeCompletionTUInfo CCTUInfo;
  
public:
  ClingTabCompletion(const CodeCompleteOptions& CodeCompleteOpts)
    : CodeCompleteConsumer(CodeCompleteOpts, false),
      CCTUInfo(new GlobalCodeCompletionAllocator) {}
  ~ClingTabCompletion() {}
  CodeCompletionAllocator &getAllocator() override { return CCTUInfo.getAllocator();}
  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

  void ProcessCodeCompleteResults(Sema &S,
                                            CodeCompletionContext Context,
                                            CodeCompletionResult *Results,
                                            unsigned NumResults) {
    std::stable_sort(Results, Results + NumResults);
    printf("printing..\n");
    printf("%d\n", NumResults);
  }

  bool Complete(Text& Line /*in+out*/,
                          size_t& Cursor /*in+out*/,
                          EditorRange& R /*out*/,
                          std::vector<std::string>& DisplayCompletions /*out*/) override {
    //Create the interpreter
    const char * const argV = "cling";
    Interpreter CodeCompletionInterp(1, &argV);;
    CodeCompletionInterp.getCI()->setCodeCompletionConsumer(this);
    CodeCompletionInterp.getCI()->getSema().CodeCompleter = this;
    //Get the results
    //CodeCompletionInterp->getCI()->getPreprocessor().SetCodeCompletionPoint();
    CodeCompletionInterp.codeComplete(Line.GetText(), Cursor);
    return true;
  }
};
}

#endif
