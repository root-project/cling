//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_USERINTERFACE_H
#define CLING_USERINTERFACE_H

#include <cstdlib>

namespace cling {
  class Interpreter;
  class MetaProcessor;

  ///\brief Makes the interpreter interactive
  ///
  class UserInterface {
  private:
    MetaProcessor* m_MetaProcessor;

    ///\brief Prints cling's startup logo
    ///
    void PrintLogo();
  public:
    UserInterface(Interpreter& interp);
    ~UserInterface();

    MetaProcessor* getMetaProcessor() { return m_MetaProcessor; }

    ///\brief Drives the interactive prompt talking to the user.
    /// @param[in] nologo - whether to show cling's welcome logo or not
    ///
    void runInteractively(bool nologo = false);
  };
}

#endif // CLING_USERINTERFACE_H
