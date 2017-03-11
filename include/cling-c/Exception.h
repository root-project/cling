//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Roman Zulak
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_CINTERACE_EXCEPTION_H
#define CLING_CINTERACE_EXCEPTION_H

#include "config.h"

// cling_ThrowCompilationException is here to keep it's declaration isolated
// from any potential RTTI issues occuring if it were located in the C++ header.
#ifdef __cplusplus
#include <string>
extern "C" void cling_ThrowCompilationException(void* UserData,
                                                const std::string& Reason,
                                                bool ShowCrashDialog);
#endif

CLING_EXTERN_C_

void* cling_ThrowIfInvalidPointer(void* Interp, void* Expr, const void* Arg);

void cling_RunLoop(bool (*RunProc)(void* Data), void* Data);

_CLING_EXTERN_C

#endif // CLING_CINTERACE_EXCEPTION_H
