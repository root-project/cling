//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Saagar Jha <saagar@saagarjha.com>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_SIGNAL_H
#define CLING_SIGNAL_H

#include <csignal>
#include <csetjmp>

extern jmp_buf before_execution;

void InterruptHandler();

#endif // CLING_SIGNAL_H
