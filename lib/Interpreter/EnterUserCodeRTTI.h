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
  InterpreterCallbacks* fCallbacks; // callbacks used to un/lock.
  void* fStateInfo = nullptr;       // info provided to ReturnedFromUserCode().
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
}

#endif // CLING_BACKENDPASSES_H
