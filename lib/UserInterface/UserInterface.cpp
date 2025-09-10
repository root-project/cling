//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/UserInterface/UserInterface.h"

#include "cling/Interpreter/Exception.h"
#include "cling/MetaProcessor/MetaProcessor.h"
#include "cling/Utils/Output.h"
#include "textinput/Callbacks.h"
#include "textinput/History.h"
#include "textinput/StreamReader.h"
#include "textinput/TerminalDisplay.h"
#include "textinput/TextInput.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/CompilerInstance.h"

namespace {
  ///\brief Class that specialises the textinput TabCompletion to allow Cling
  /// to code complete through its own textinput mechanism which is part of the
  /// UserInterface.
  ///
  class UITabCompletion : public textinput::TabCompletion {
    const cling::Interpreter& m_ParentInterpreter;

  public:
    UITabCompletion(const cling::Interpreter& Parent) :
                    m_ParentInterpreter(Parent) {}
    ~UITabCompletion() {}

    bool Complete(textinput::Text& Line /*in+out*/,
                  size_t& Cursor /*in+out*/,
                  textinput::EditorRange& R /*out*/,
                  std::vector<std::string>& Completions /*out*/) override {
      m_ParentInterpreter.codeComplete(Line.GetText(), Cursor, Completions);
      return true;
    }
  };

  ///\brief Delays ~TextInput until after ~StreamReader and ~TerminalDisplay
  ///
  class TextInputHolder {
    textinput::StreamReader* m_Reader;
    textinput::TerminalDisplay* m_Display;
    textinput::TextInput m_Input;

  public:
    TextInputHolder(llvm::SmallString<512>& Hist)
        : m_Reader(textinput::StreamReader::Create()),
          m_Display(textinput::TerminalDisplay::Create()),
          m_Input(*m_Reader, *m_Display, Hist.empty() ? 0 : Hist.c_str()) {}

    ~TextInputHolder() {
      delete m_Reader;
      delete m_Display;
    }

    textinput::TextInput* operator -> () { return &m_Input; }
  };

  llvm::SmallString<512> GetHistoryFilePath() {
    if (std::getenv("CLING_NOHISTORY")) {
      return {};
    }

    if (const char* HistFileEnvvar = std::getenv("CLING_HISTFILE")) {
      return llvm::StringRef{HistFileEnvvar};
    }

    // History file search order according to XDG Base Directory Specification:
    //
    // ${XDG_STATE_HOME}/cling/history
    // ~/.local/state/cling/history
    // ~/.cling_history
    const char* StateHome = std::getenv("XDG_STATE_HOME");

    if (!StateHome) {
      StateHome = "~/.local/state";
    }

    llvm::SmallString<512> FilePath;

    if (!llvm::sys::fs::real_path(StateHome, FilePath, true)) {
      // If xdg state home directory exists then create cling subdirectory if
      // the latter does not exist.
      if (llvm::sys::fs::is_directory(FilePath) &&
          !llvm::sys::fs::create_directory(FilePath += "/cling")) {
        return FilePath += "/history";
      }
    }

    if (llvm::sys::path::home_directory(FilePath)) {
      return FilePath += "/.cling_history";
    }

    cling::errs() << "Failed to create command history file\n";
    return {};
  }
}

namespace cling {

  UserInterface::UserInterface(Interpreter& interp) {
    m_MetaProcessor.reset(new MetaProcessor(interp, cling::outs()));
    llvm::install_fatal_error_handler(&CompilationException::throwingHandler);
  }

  UserInterface::~UserInterface() {}

  void UserInterface::runInteractively(bool nologo /* = false */) {
    if (!nologo) {
      PrintLogo();
    }

    auto histfilePath{GetHistoryFilePath()};

    const auto Completion =
        std::make_unique<UITabCompletion>(m_MetaProcessor->getInterpreter());
    TextInputHolder TI(histfilePath);

    TI->SetCompletion(Completion.get());

    if (const char* HistSizeEnvvar = std::getenv("CLING_HISTSIZE")) {
      const size_t HistSize = std::strtoull(HistSizeEnvvar, nullptr, 0);

      // std::strtoull() returns 0 if the parsing fails.
      // zero HistSize will disable history logging to file.
      // refer to textinput::History::AppendToFile()
      TI->SetHistoryMaxDepth(HistSize);
      TI->SetHistoryPruneLength(
          static_cast<size_t>(textinput::History::kPruneLengthDefault));
    }

    bool Done = false;
    std::string Line;
    std::string Prompt("[cling]$ ");

    while (!Done) {
      try {
        m_MetaProcessor->getOuts().flush();
        {
          MetaProcessor::MaybeRedirectOutputRAII RAII(*m_MetaProcessor);
          TI->SetPrompt(Prompt.c_str());
          Done = TI->ReadInput() == textinput::TextInput::kRREOF;
          TI->TakeInput(Line);
          if (Done && Line.empty())
            break;
        }

        cling::Interpreter::CompilationResult compRes;
        const int indent = m_MetaProcessor->process(Line, compRes);

        // Quit requested?
        if (indent < 0)
          break;

        Prompt.replace(7, std::string::npos,
           m_MetaProcessor->getInterpreter().isRawInputEnabled() ? "! " : "$ ");

        // Continuation requested?
        if (indent > 0) {
          Prompt.append(1, '?');
          Prompt.append(indent * 3, ' ');
        }
      }
      catch(InterpreterException& e) {
        if (!e.diagnose()) {
          cling::errs() << ">>> Caught an interpreter exception!\n"
                        << ">>> " << e.what() << '\n';
        }
      }
      catch(std::exception& e) {
        cling::errs() << ">>> Caught a std::exception!\n"
                     << ">>> " << e.what() << '\n';
      }
      catch(...) {
        cling::errs() << "Exception occurred. Recovering...\n";
      }
    }
    m_MetaProcessor->getOuts().flush();
  }

  void UserInterface::PrintLogo() {
    llvm::raw_ostream& outs = m_MetaProcessor->getOuts();
    const clang::LangOptions& LangOpts
      = m_MetaProcessor->getInterpreter().getCI()->getLangOpts();
    if (LangOpts.CPlusPlus) {
      outs << "\n"
        "****************** CLING ******************\n"
        "* Type C++ code and press enter to run it *\n"
        "*     Type .? for help and .q to exit     *\n"
        "*******************************************\n";
    } else {
      outs << "\n"
        "***************** CLING *****************\n"
        "* Type C code and press enter to run it *\n"
        "*    Type .? for help and .q to exit    *\n"
        "*****************************************\n";
    }
  }
} // end namespace cling
