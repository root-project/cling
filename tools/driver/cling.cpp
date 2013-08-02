//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/StoredValueRef.h"
#include "cling/MetaProcessor/MetaProcessor.h"
#include "cling/UserInterface/UserInterface.h"

#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/Support/Signals.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ManagedStatic.h"

#include <iostream>
#include <vector>
#include <string>

int main( int argc, char **argv ) {

  llvm::llvm_shutdown_obj shutdownTrigger;

  //llvm::sys::PrintStackTraceOnErrorSignal();
  //llvm::PrettyStackTraceProgram X(argc, argv);

  // Set up the interpreter
  cling::Interpreter interp(argc, argv);
  if (interp.getOptions().Help) {
    return 0;
  }

  clang::CompilerInstance* CI = interp.getCI();
  interp.AddIncludePath(".");

  for (size_t I = 0, N = interp.getOptions().LibsToLoad.size(); I < N; ++I) {
    interp.loadFile(interp.getOptions().LibsToLoad[I]);
  }

   bool ret = true;
   const std::vector<clang::FrontendInputFile>& Inputs
     = CI->getInvocation().getFrontendOpts().Inputs;

   // Interactive means no input (or one input that's "-")
   bool Interactive = Inputs.empty() || (Inputs.size() == 1
                                         && Inputs[0].getFile() == "-");

   cling::UserInterface ui(interp);
   // If we are not interactive we're supposed to parse files
   if (!Interactive) {
     for (size_t I = 0, N = Inputs.size(); I < N; ++I) {
       std::string line(".x ");
       line += Inputs[I].getFile();
       cling::Interpreter::CompilationResult compRes;
       ui.getMetaProcessor()->process(line.c_str(), compRes, 0);
       ret = !CI->getDiagnostics().getClient()->getNumErrors();
     }
   }
   else {
      ui.runInteractively(interp.getOptions().NoLogo);
   }
   

   // if we are running with -verify a reported has to be returned as unsuccess.
   // This is relevant especially for the test suite.
   if (CI->getDiagnosticOpts().VerifyDiagnostics) {
     // If there was an error that came from the verifier we must return 1 as
     // an exit code for the process. This will make the test fail as expected.
     clang::DiagnosticConsumer* client = CI->getDiagnostics().getClient();
     client->EndSourceFile();
     ret = client->getNumErrors();

     // The interpreter expects BeginSourceFile/EndSourceFiles to be balanced.
     client->BeginSourceFile(CI->getLangOpts(), &CI->getPreprocessor());
   }

   return ret;
}
