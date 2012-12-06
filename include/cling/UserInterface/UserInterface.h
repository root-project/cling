//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef CLING_USERINTERFACE_H
#define CLING_USERINTERFACE_H

#include "llvm/ADT/OwningPtr.h"

namespace cling {
  class Interpreter;
  class MetaProcessor;

  ///\brief Makes the interpreter interactive
  ///
  class UserInterface {
  private:
    llvm::OwningPtr<MetaProcessor> m_MetaProcessor;

    ///\brief Prints cling's startup logo
    ///
    void PrintLogo();
  public:
    UserInterface(Interpreter& interp);
    ~UserInterface();

    MetaProcessor* getMetaProcessor() { return m_MetaProcessor.get(); }

    ///\brief Drives the interactive prompt talking to the user.
    /// @param[in] nologo - whether to show cling's welcome logo or not
    ///
    void runInteractively(bool nologo = false);
  };
}

#endif // CLING_USERINTERFACE_H
