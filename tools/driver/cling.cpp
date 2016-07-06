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
#include "clang/FrontendTool/Utils.h"

#include "llvm/Support/Signals.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ManagedStatic.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#if defined(WIN32) && defined(_MSC_VER)
#include <crtdbg.h>
#endif

// If we are running with -verify a reported has to be returned as unsuccess.
// This is relevant especially for the test suite.
static int checkDiagErrors(clang::CompilerInstance* CI, unsigned* OutErrs = 0) {

  unsigned Errs = CI->getDiagnostics().getClient()->getNumErrors();

  if (CI->getDiagnosticOpts().VerifyDiagnostics) {
    // If there was an error that came from the verifier we must return 1 as
    // an exit code for the process. This will make the test fail as expected.
    clang::DiagnosticConsumer* Client = CI->getDiagnostics().getClient();
    Client->EndSourceFile();
    Errs = Client->getNumErrors();

    // The interpreter expects BeginSourceFile/EndSourceFiles to be balanced.
    Client->BeginSourceFile(CI->getLangOpts(), &CI->getPreprocessor());
  }

  if (OutErrs)
    *OutErrs = Errs;

  return Errs ? EXIT_FAILURE : EXIT_SUCCESS;
}


int main( int argc, char **argv ) {

  llvm::llvm_shutdown_obj shutdownTrigger;

  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);

#if defined(_WIN32) && defined(_MSC_VER)
  // Suppress error dialogs to avoid hangs on build nodes.
  // One can use an environment variable (Cling_GuiOnAssert) to enable
  // the error dialogs.
  const char *EnablePopups = getenv("Cling_GuiOnAssert");
  if (EnablePopups == nullptr || EnablePopups[0] == '0') {
    ::_set_error_mode(_OUT_TO_STDERR);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  }
#endif

  // Set up the interpreter
  cling::Interpreter interp(argc, argv);

  if (!interp.isValid()) {
    const cling::InvocationOptions& Opts = interp.getOptions();
    if (Opts.Help || Opts.ShowVersion)
      return EXIT_SUCCESS;

    unsigned ErrsReported = 0;
    if (clang::CompilerInstance* CI = interp.getCIOrNull()) {
      // If output requested and execution succeeded let the DiagnosticsEngine
      // determine the result code
      if (Opts.CompilerOpts.HasOutput && ExecuteCompilerInvocation(CI))
        return checkDiagErrors(CI);

      checkDiagErrors(CI, &ErrsReported);
    }

    // If no errors have been reported, try perror
    if (ErrsReported == 0)
      ::perror("Could not create Interpreter instance");

    return EXIT_FAILURE;
  }

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

  // Only for test/OutputRedirect.C, but shouldn't affect performance too much.
  ::fflush(stdout);
  ::fflush(stderr);

  return checkDiagErrors(interp.getCI());
}
