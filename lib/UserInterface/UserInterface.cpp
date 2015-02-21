//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/UserInterface/UserInterface.h"

#include "cling/UserInterface/CompilationException.h"
#include "cling/Interpreter/RuntimeException.h"
#include "cling/MetaProcessor/MetaProcessor.h"
#include "textinput/TextInput.h"
#include "textinput/StreamReader.h"
#include "textinput/TerminalDisplay.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Config/config.h"

#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/CompilerInstance.h"

// Fragment copied from LLVM's raw_ostream.cpp
#if defined(HAVE_UNISTD_H)
# include <unistd.h>
#endif

#if defined(LLVM_ON_WIN32)
#include <Shlobj.h>
#endif

#if defined(_MSC_VER)
#ifndef STDIN_FILENO
# define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
# define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
# define STDERR_FILENO 2
#endif
#endif

#include <memory>

namespace {
  // Handle fatal llvm errors by throwing an exception.
  // Yes, throwing exceptions in error handlers is bad.
  // Doing nothing is pretty terrible, too.
  void exceptionErrorHandler(void * /*user_data*/,
                             const std::string& reason,
                             bool /*gen_crash_diag*/) {
    throw cling::CompilationException(reason);
  }
#if defined(LLVM_ON_UNIX)
  static void GetUserHomeDirectory(llvm::SmallVectorImpl<char>& str) {
    str.clear();
    const char* home = getenv("HOME");
    if (!home)
      home = "/";
    llvm::StringRef SRhome(home);
    str.insert(str.begin(), SRhome.begin(), SRhome.end());
  }
#elif defined(LLVM_ON_WIN32)
  static void GetUserHomeDirectory(llvm::SmallVectorImpl<char>& str) {
    str.reserve(MAX_PATH);
    HRESULT res = SHGetFolderPathA(NULL,
                                   CSIDL_FLAG_CREATE | CSIDL_APPDATA,
                                   NULL,
                                   SHGFP_TYPE_CURRENT,
                                   str.data());
    if (res != S_OK) {
      assert(0 && "Failed to get user home directory");
      llvm::StringRef SRhome("\\");
      str.insert(str.begin(), SRhome.begin(), SRhome.end());
    }
  }
#else
# error "Unsupported platform."
#endif
}

namespace cling {
  // Declared in CompilationException.h; vtable pinned here.
  CompilationException::~CompilationException() throw() {}

  UserInterface::UserInterface(Interpreter& interp) {
    // We need stream that doesn't close its file descriptor, thus we are not
    // using llvm::outs. Keeping file descriptor open we will be able to use
    // the results in pipes (Savannah #99234).
    static llvm::raw_fd_ostream m_MPOuts (STDOUT_FILENO, /*ShouldClose*/false);
    m_MetaProcessor.reset(new MetaProcessor(interp, m_MPOuts));
    llvm::install_fatal_error_handler(&exceptionErrorHandler);
  }

  UserInterface::~UserInterface() {}

  void UserInterface::runInteractively(bool nologo /* = false */) {
    if (!nologo) {
      PrintLogo();
    }

    llvm::SmallString<512> histfilePath;
    if (!getenv("CLING_NOHISTORY")) {
      // History file is $HOME/.cling_history
      static const char* histfile = ".cling_history";
      GetUserHomeDirectory(histfilePath);
      llvm::sys::path::append(histfilePath, histfile);
    }

    using namespace textinput;
    std::unique_ptr<StreamReader> R(StreamReader::Create());
    std::unique_ptr<TerminalDisplay> D(TerminalDisplay::Create());
    TextInput TI(*R, *D, histfilePath.empty() ? 0 : histfilePath.c_str());

    TI.SetPrompt("[cling]$ ");
    std::string line;

    jmp_buf env;
    int val = setjmp(env);
    if (!val) {
      Interpreter::getNullDerefJump() = &env;
    } else {
      llvm::errs() << "LongJmp occurred. Recovering...\n";
    }
    while (true) {
      try {
        m_MetaProcessor->getOuts().flush();
        TextInput::EReadResult RR = TI.ReadInput();
        TI.TakeInput(line);
        if (RR == TextInput::kRREOF) {
          break;
        }

        cling::Interpreter::CompilationResult compRes;
        MetaProcessor::MaybeRedirectOutputRAII RAII(m_MetaProcessor.get());
        int indent
          = m_MetaProcessor->process(line.c_str(), compRes, 0/*result*/);
        // Quit requested
        if (indent < 0)
          break;
        std::string Prompt = "[cling]";
        if (m_MetaProcessor->getInterpreter().isRawInputEnabled())
          Prompt.append("! ");
        else
          Prompt.append("$ ");

        if (indent > 0)
          // Continuation requested.
          Prompt.append('?' + std::string(indent * 3, ' '));

        TI.SetPrompt(Prompt.c_str());

      }
      catch(runtime::NullDerefException& e) {
        e.diagnose();
      }
      catch(runtime::InterpreterException& e) {
        llvm::errs() << ">>> Caught an interpreter exception!\n"
                     << ">>> " << e.what() << '\n';
      }
      catch(std::exception& e) {
        llvm::errs() << ">>> Caught a std::exception!\n"
                     << ">>> " << e.what() << '\n';
      }
      catch(...) {
        llvm::errs() << "Exception occurred. Recovering...\n";
      }
    }
  }

  void UserInterface::PrintLogo() {
    llvm::raw_ostream& outs = m_MetaProcessor->getOuts();
    const clang::LangOptions& LangOpts
      = m_MetaProcessor->getInterpreter().getCI()->getLangOpts();
    if (LangOpts.CPlusPlus) {
      outs << "\n"
        "****************** CLING ******************\n"
        "* Type C++ code and press enter to run it *\n"
        "*             Type .q to exit             *\n"
        "*******************************************\n";
    } else {
      outs << "\n"
        "***************** CLING *****************\n"
        "* Type C code and press enter to run it *\n"
        "*            Type .q to exit            *\n"
        "*****************************************\n";
    }
  }
} // end namespace cling













