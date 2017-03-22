//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/UserInterface/UserInterface.h"

#include "cling/MetaProcessor/MetaProcessor.h"
#include "cling/Utils/Output.h"
#include "cling-c/Exception.h"
#include "textinput/Callbacks.h"
#include "textinput/TextInput.h"
#include "textinput/StreamReader.h"
#include "textinput/TerminalDisplay.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Config/config.h"

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

  static bool Quit(cling::MetaProcessor& MP) {
    MP.getOuts().flush();
    return false;
  }
}

namespace cling {

  UserInterface::UserInterface(Interpreter& Interp, const char* Prompt)
      : m_Prompt(Prompt), m_PromptLen(m_Prompt.size()) {
    m_MetaProcessor.reset(new MetaProcessor(Interp, cling::outs()));
    llvm::install_fatal_error_handler(&cling_ThrowCompilationException);
  }

  UserInterface::~UserInterface() {}

  struct UserInterface::TextInput {
    std::unique_ptr<textinput::StreamReader> Reader;
    std::unique_ptr<textinput::TerminalDisplay> Display;
    std::unique_ptr<textinput::TextInput> Input;

    TextInput(const cling::Interpreter& Interp)
        : Reader(textinput::StreamReader::Create()),
          Display(textinput::TerminalDisplay::Create()) {
      llvm::SmallString<512> History;
      if (!getenv("CLING_NOHISTORY")) {
        // History file is $HOME/.cling_history
        if (llvm::sys::path::home_directory(History))
          llvm::sys::path::append(History, ".cling_history");
      }

      Input.reset(new textinput::TextInput(
          *Reader, *Display, History.empty() ? nullptr : History.c_str()));

      Input->SetCompletion(new UITabCompletion(Interp));
    }
    
    static bool RunLoop(void* This) {
      return reinterpret_cast<UserInterface*>(This)->RunLoop();
    }
  };

  bool UserInterface::RunLoop() {
    MetaProcessor& MP = *m_MetaProcessor;
    textinput::TextInput& TI = *m_TextInput->Input;
    MP.getOuts().flush();

    std::string Line;
    {
      cling::MetaProcessor::MaybeRedirectOutputRAII RAII(MP);
      TI.SetPrompt(m_Prompt.c_str());
      if (TI.ReadInput() == textinput::TextInput::kRREOF)
        return Quit(MP);
      TI.TakeInput(Line);
    }

    cling::Interpreter::CompilationResult Result;
    const int Indent = MP.process(Line.c_str(), Result);

    // Quit requested?
    if (Indent < 0)
      return Quit(MP);

    m_Prompt.replace(m_PromptLen, std::string::npos,
                     MP.getInterpreter().isRawInputEnabled() ? "! " : "$ ");

    // Continuation requested?
    if (Indent > 0) {
      m_Prompt.append(1, '?');
      m_Prompt.append(Indent * 3, ' ');
    }
    return true;
  }

  void UserInterface::RunInteractively() {
    if (!m_MetaProcessor->getInterpreter().getOptions().NoLogo)
      PrintLogo();

    m_Prompt.append("$ ");
    m_TextInput.reset(new TextInput(m_MetaProcessor->getInterpreter()));

    cling_RunLoop(&TextInput::RunLoop, reinterpret_cast<void*>(this));
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
