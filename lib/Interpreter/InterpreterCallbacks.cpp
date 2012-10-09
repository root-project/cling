//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Vassil Vassilev <vvasilev@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/InterpreterCallbacks.h"

#include "cling/Interpreter/Interpreter.h"

#include "clang/Sema/Sema.h"

namespace cling {

  // pin the vtable here
  InterpreterExternalSemaSource::~InterpreterExternalSemaSource() {}

  bool InterpreterExternalSemaSource::LookupUnqualified(clang::LookupResult& R, 
                                                        clang::Scope* S) {
    if (m_Callbacks && m_Callbacks->isEnabled())
      return m_Callbacks->LookupObject(R, S);
    
    return false;
  }

  InterpreterCallbacks::InterpreterCallbacks(Interpreter* interp, bool enabled)
    : m_Interpreter(interp), m_Enabled(enabled) {
    m_SemaExternalSource.reset(new InterpreterExternalSemaSource(this));
    m_Interpreter->getSema().addExternalSource(m_SemaExternalSource.get());
  }


  // pin the vtable here
  InterpreterCallbacks::~InterpreterCallbacks() {}

  bool InterpreterCallbacks::LookupObject(clang::LookupResult&, clang::Scope*) {
    return false;
  }

} // end namespace cling
