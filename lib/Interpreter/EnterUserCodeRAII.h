//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_ENTERUSERCODERAII_H
#define CLING_ENTERUSERCODERAII_H

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"

namespace cling {
///\brief Unlocks and then upon destruction locks the interpreter again.
struct EnterUserCodeRAII {
  /// Callbacks used to un/lock.
  InterpreterCallbacks* fCallbacks;
  /// Info provided to ReturnedFromUserCode().
  void* fStateInfo = nullptr;
  EnterUserCodeRAII(InterpreterCallbacks* callbacks): fCallbacks(callbacks) {
    if (fCallbacks)
      fStateInfo = fCallbacks->EnteringUserCode();
  }

  EnterUserCodeRAII(Interpreter& interp):
    EnterUserCodeRAII(interp.getCallbacks()) {}

  ~EnterUserCodeRAII() {
    if (fCallbacks)
      fCallbacks->ReturnedFromUserCode(fStateInfo);
  }
};

struct LockCompilationDuringUserCodeExecutionRAII {
  /// Callbacks used to un/lock.
  InterpreterCallbacks* fCallbacks;
  /// Info provided to UnlockCompilationDuringUserCodeExecution().
  void* fStateInfo = nullptr;
  LockCompilationDuringUserCodeExecutionRAII(InterpreterCallbacks* callbacks):
  fCallbacks(callbacks) {
    if (fCallbacks)
      fStateInfo = fCallbacks->LockCompilationDuringUserCodeExecution();
  }

  LockCompilationDuringUserCodeExecutionRAII(Interpreter& interp):
  LockCompilationDuringUserCodeExecutionRAII(interp.getCallbacks()) {}

  ~LockCompilationDuringUserCodeExecutionRAII() {
    if (fCallbacks)
      fCallbacks->UnlockCompilationDuringUserCodeExecution(fStateInfo);
  }
};
}

#endif // CLING_ENTERUSERCODERAII_H
