#ifndef CLINGTABCOMPLETION_H
#define CLINGTABCOMPLETION_H

#include "cling/Interpreter/Interpreter.h"
#include "ClingCodeCompleteConsumer.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"

#include "textinput/Text.h"
#include "textinput/Editor.h"
#include "textinput/Callbacks.h"

namespace textinput{
class ClingTabCompletion : public textinput::TabCompletion {
  const cling::Interpreter& ParentInterp;
  
public:
  ClingTabCompletion(const cling::Interpreter& Parent) : ParentInterp(Parent) {}

  ~ClingTabCompletion() {}

  bool Complete(Text& Line /*in+out*/,
                size_t& Cursor /*in+out*/,
                EditorRange& R /*out*/,
                std::vector<std::string>& DisplayCompletions /*out*/) override {
    //Get the results
    const char * const argV = "cling";
    cling::Interpreter CodeCompletionInterp(ParentInterp, 1, &argV);

    // CreateCodeCompleteConsumer with InterpreterCallbacks
    //interpretercallbacks -> cerateCodeCompleteConsumer()
    clang::PrintingCodeCompleteConsumer* consumer = 
                          new clang::PrintingCodeCompleteConsumer(
                            ParentInterp.getCI()->getFrontendOpts().CodeCompleteOpts,
                            llvm::outs());
    clang::CompilerInstance* codeCompletionCI = CodeCompletionInterp.getCI();
    // codeCompletionCI will own consumer!
    codeCompletionCI->setCodeCompletionConsumer(consumer);
    clang::Sema& codeCompletionSemaRef = codeCompletionCI->getSema();
    codeCompletionSemaRef.CodeCompleter = consumer;

    // Ignore diagnostics when we tab complete
    clang::IgnoringDiagConsumer* ignoringDiagConsumer = new clang::IgnoringDiagConsumer();
    codeCompletionSemaRef.getDiagnostics().setClient(ignoringDiagConsumer, true);

    auto Owner = ParentInterp.getCI()->getSema().getDiagnostics().takeClient();
    auto Client = ParentInterp.getCI()->getSema().getDiagnostics().getClient();
    ParentInterp.getCI()->getSema().getDiagnostics().setClient(ignoringDiagConsumer, false);
    CodeCompletionInterp.codeComplete(Line.GetText(), Cursor);
  
    // // Reset the original diag client for parent interpreter
    ParentInterp.getCI()->getSema().getDiagnostics().setClient(Client, Owner.release() != nullptr);
    // FIX-ME : Change it in the Incremental Parser
    //CodeCompletionInterp.unload(1);
    
  return true;
  }
};
}

#endif
