//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "cling/UserInterface/UserInterface.h"

#include "cling/MetaProcessor/MetaProcessor.h"
#include "textinput/TextInput.h"
#include "textinput/StreamReader.h"
#include "textinput/TerminalDisplay.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/PathV1.h"

namespace cling {
  UserInterface::UserInterface(Interpreter& interp) {
    // We need stream that doesn't close its file descriptor, thus we are not
    // using llvm::outs. Keeping file descriptor open we will be able to use
    // the results in pipes (Savannah #99234).
    static llvm::raw_fd_ostream m_MPOuts (STDOUT_FILENO, /*ShouldClose*/false);
    m_MetaProcessor.reset(new MetaProcessor(interp, m_MPOuts));
  }

  UserInterface::~UserInterface() {}

  void UserInterface::runInteractively(bool nologo /* = false */) {
    if (!nologo) {
      PrintLogo();
    }

    // History file is $HOME/.cling_history
    static const char* histfile = ".cling_history";
    llvm::sys::Path histfilePath = llvm::sys::Path::GetUserHomeDirectory();
    histfilePath.appendComponent(histfile);

    using namespace textinput;
    StreamReader* R = StreamReader::Create();
    TerminalDisplay* D = TerminalDisplay::Create();
    TextInput TI(*R, *D, histfilePath.c_str());

    TI.SetPrompt("[cling]$ ");
    std::string line;

    while (true) {
      m_MetaProcessor->getOuts().flush();
      TextInput::EReadResult RR = TI.ReadInput();
      TI.TakeInput(line);
      if (RR == TextInput::kRREOF) {
        break;
      }
     
      int indent = m_MetaProcessor->process(line.c_str());
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
  }

  void UserInterface::PrintLogo() {
    llvm::raw_ostream& outs = m_MetaProcessor->getOuts();
    outs << "\n";
    outs << "****************** CLING ******************" << "\n";
    outs << "* Type C++ code and press enter to run it *" << "\n";
    outs << "*             Type .q to exit             *" << "\n";
    outs << "*******************************************" << "\n";
  }
} // end namespace cling
