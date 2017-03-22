//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Roman Zulak
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_CINTERACE_CONFIG_H
#define CLING_CINTERACE_CONFIG_H

#ifdef __cplusplus
#define CLING_EXTERN_C_ extern "C" {
#define _CLING_EXTERN_C }
#else
#define CLING_EXTERN_C_
#define _CLING_EXTERN_C
#endif

#endif // CLING_CINTERACE_CONFIG_H
