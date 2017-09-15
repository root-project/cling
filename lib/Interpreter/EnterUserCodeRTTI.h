//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_ENTERUSERCODERTTI_H
#define CLING_ENTERUSERCODERTTI_H

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"

namespace cling {
///\brief Unlocks and then upon destruction locks the interpreter again.
struct EnterUserCodeRTTI {
  /// Callbacks used to un/lock.
  InterpreterCallbacks* fCallbacks;
  /// Info provided to ReturnedFromUserCode().
  void* fStateInfo = nullptr;
  EnterUserCodeRTTI(InterpreterCallbacks* callbacks): fCallbacks(callbacks) {
    if (fCallbacks)
      fStateInfo = fCallbacks->EnteringUserCode();
  }

  EnterUserCodeRTTI(Interpreter& interp):
    EnterUserCodeRTTI(interp.getCallbacks()) {}

  ~EnterUserCodeRTTI() {
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

#endif // CLING_BACKENDPASSES_H
