//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/Interpreter.h"
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
       ui.getMetaProcessor()->process(line.c_str());
       ret = !CI->getDiagnostics().getClient()->getNumErrors();
     }
   }
   else {
      cling::UserInterface ui(interp);
      ui.runInteractively(interp.getOptions().NoLogo);
   }

   // if we are running with -verify a reported has to be returned as unsuccess.
   // This is relevan especially for the test suite.
   if (CI->getDiagnosticOpts().VerifyDiagnostics)
     ret = !CI->getDiagnostics().getClient()->getNumErrors();

   return ret ? 0 : 1;
}
