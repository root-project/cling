//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Lukasz Janyst <ljanyst@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
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
#include <fstream>
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


  // Interactive means no input (or one input that's "-")
  std::vector<std::string>& Inputs = interp.getOptions().Inputs;
  bool Interactive = Inputs.empty() || (Inputs.size() == 1
                                        && Inputs[0] == "-");

  cling::UserInterface ui(interp);
  // If we are not interactive we're supposed to parse files
  if (!Interactive) {
    for (size_t I = 0, N = Inputs.size(); I < N; ++I) {
      std::string cmd;
      cling::Interpreter::CompilationResult compRes;
      if (!interp.lookupFileOrLibrary(Inputs[I]).empty()) {
        std::ifstream infile(interp.lookupFileOrLibrary(Inputs[I]));
        std::string line;
        std::getline(infile, line);
        if (line[0] == '#' && line[1] == '!') {
          // TODO: Check whether the filename specified after #! is the current
          // executable.
          while(std::getline(infile, line)) {
            ui.getMetaProcessor()->process(line.c_str(), compRes, 0);
          }
          continue;
        }
        else
          cmd += ".x ";
      }
      cmd += Inputs[I];
      ui.getMetaProcessor()->process(cmd.c_str(), compRes, 0);
    }
  }
  else {
    ui.runInteractively(interp.getOptions().NoLogo);
  }

  bool ret = CI->getDiagnostics().getClient()->getNumErrors();

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
