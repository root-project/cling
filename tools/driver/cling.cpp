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
  cling::Interpreter Interp(argc, argv);
  const cling::InvocationOptions& Opts = Interp.getOptions();

  if (!Interp.isValid()) {
    if (Opts.Help || Opts.ShowVersion)
      return EXIT_SUCCESS;

    using namespace cling;
    const Interpreter::CompilationResult ExecRes = Interp.executeInvocation();
    if (ExecRes != Interpreter::kFailure) {
      const int ErrsReported = Interp.verifyDiagnostics();

      // If output succeeded and diagnostics verified, we are done.
      if (ExecRes == Interpreter::kSuccess && !ErrsReported)
        return EXIT_SUCCESS;

      // FIXME: This info is barely useful
      if (ErrsReported <= 0)
        ::fprintf(stderr, "Unknown error");
    } else
      ::perror("Could not create Interpreter instance");

    return EXIT_FAILURE;
  }

  Interp.AddIncludePath(".");

  for (const std::string& Lib : Opts.LibsToLoad)
    Interp.loadFile(Lib);

  cling::UserInterface Ui(Interp);
  // If we are not interactive we're supposed to parse files
  if (!Opts.IsInteractive()) {
    for (const std::string &Input : Opts.Inputs) {
      std::string Cmd;
      cling::Interpreter::CompilationResult Result;
      const std::string Filepath = Interp.lookupFileOrLibrary(Input);
      if (!Filepath.empty()) {
        std::ifstream File(Filepath);
        std::string Line;
        std::getline(File, Line);
        if (Line[0] == '#' && Line[1] == '!') {
          // TODO: Check whether the filename specified after #! is the current
          // executable.
          while (std::getline(File, Line)) {
            Ui.getMetaProcessor()->process(Line, Result, 0);
          }
          continue;
        }
        Cmd += ".x ";
      }
      Cmd += Input;
      Ui.getMetaProcessor()->process(Cmd, Result, 0);
    }
  }
  else {
    Ui.runInteractively(Opts.NoLogo);
  }

  // Only for test/OutputRedirect.C, but shouldn't affect performance too much.
  ::fflush(stdout);
  ::fflush(stderr);

  return Interp.verifyDiagnostics() ? EXIT_FAILURE : EXIT_SUCCESS;
}
