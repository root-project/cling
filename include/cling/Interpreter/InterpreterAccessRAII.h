//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_INTERPRETERACCESSRAII_H
#define CLING_INTERPRETERACCESSRAII_H

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"

namespace cling {
///\brief Locks and unlocks access to the interpreter.
struct InterpreterAccessRAII {
  /// Callbacks used to un/lock.
  InterpreterCallbacks* fCallbacks;
  /// Info provided to UnlockCompilationDuringUserCodeExecution().
  void* fStateInfo = nullptr;

  InterpreterAccessRAII(InterpreterCallbacks* callbacks):
    fCallbacks(callbacks)
  {
    if (fCallbacks)
      // The return value is alway a nullptr.
      fStateInfo = fCallbacks->LockCompilationDuringUserCodeExecution();
  }

  InterpreterAccessRAII(Interpreter& interp):
    InterpreterAccessRAII(interp.getCallbacks()) {}

  ~InterpreterAccessRAII()
  {
    if (fCallbacks)
      fCallbacks->UnlockCompilationDuringUserCodeExecution(fStateInfo);
  }
};
}

#endif // CLING_ENTERUSERCODERAII_H
