//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Lukasz Janyst <ljanyst@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_USERINTERFACE_H
#define CLING_USERINTERFACE_H

#include <memory>
#include <string>

namespace cling {
  class Interpreter;
  class MetaProcessor;

  ///\brief Makes the interpreter interactive
  ///
  class UserInterface {
    struct TextInput;
    std::unique_ptr<MetaProcessor> m_MetaProcessor;
    std::unique_ptr<TextInput> m_TextInput;
    std::string m_Prompt;
    const size_t m_PromptLen;

    ///\brief Prints cling's startup logo
    ///
    void PrintLogo();

    bool RunLoop();

  public:
    UserInterface(Interpreter& Interp, const char* Prompt = "[cling]");
    ~UserInterface();

    MetaProcessor* getMetaProcessor() { return m_MetaProcessor.get(); }

    ///\brief Drives the interactive prompt talking to the user.
    /// @param[in] nologo - whether to show cling's welcome logo or not
    ///
    void RunInteractively();
  };
}

#endif // CLING_USERINTERFACE_H
