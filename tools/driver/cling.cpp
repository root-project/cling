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

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

static llvm::SmallString<512> getConfigDirPath() {
  llvm::SmallString<512> DirPath{
      std::getenv("CLING_HOME")}; // nullptr is also fine

  if (!DirPath.empty()) {
    return DirPath;
  }

  if (llvm::sys::path::user_config_directory(DirPath)) {
    llvm::sys::path::append(DirPath, "cling");
    return DirPath;
  }

  if (llvm::sys::path::home_directory(DirPath)) {
    return DirPath;
  }

  return {};
}

static void runStartupFiles(cling::UserInterface& Ui) {
  llvm::SmallString<512> StartupFilesDir{getConfigDirPath()};

  if (StartupFilesDir.empty()) {
    return;
  }

  llvm::sys::path::append(StartupFilesDir, ".cling.d");

  std::vector<std::string> FilePaths;
  std::error_code EC;

  for (llvm::sys::fs::directory_iterator DirIt(StartupFilesDir, EC), DirEnd;
       DirIt != DirEnd && !EC; DirIt.increment(EC)) {
    if (DirIt->type() == llvm::sys::fs::file_type::regular_file &&
        llvm::sys::path::extension(DirIt->path()) == ".C") {
      FilePaths.emplace_back(DirIt->path());
    }
  }

  std::sort(FilePaths.begin(), FilePaths.end());

  for (const auto& File : FilePaths) {
    auto Result{cling::Interpreter::CompilationResult::kSuccess};

    Ui.getMetaProcessor()->process(".x " + File, Result, nullptr);

    if (Result != cling::Interpreter::CompilationResult::kSuccess) {
      std::cerr << "Error running startup file " << File << '\n';
    }
  }
}

int main( int argc, char **argv ) {

  llvm::llvm_shutdown_obj shutdownTrigger;

  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);

#if defined(_WIN32) && defined(_MSC_VER)
  // Suppress error dialogs to avoid hangs on build nodes.
  // One can use an environment variable (Cling_GuiOnAssert) to enable
  // the error dialogs.
  const char *EnablePopups = std::getenv("Cling_GuiOnAssert");
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

    unsigned ErrsReported = 0;
    if (clang::CompilerInstance* CI = Interp.getCIOrNull()) {
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

  Interp.AddIncludePath(".");

  for (const std::string& Lib : Opts.LibsToLoad)
    Interp.loadFile(Lib);

  cling::UserInterface Ui(Interp);

  runStartupFiles(Ui);

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

  return checkDiagErrors(Interp.getCI());
}
