//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Saagar Jha <saagar@saagarjha.com>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Signal.h"

#include "llvm/Support/Signals.h"

jmp_buf before_execution;

void InterruptHandler() {
  // LLVM automatically resets the handler after every call. Set it back to our
  // handler for the next time we get a SIGINT.
  llvm::sys::SetInterruptFunction(InterruptHandler);
  longjmp(before_execution, 1);
}
