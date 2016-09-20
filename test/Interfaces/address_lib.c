/*------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//----------------------------------------------------------------------------*/

// RUN: true
// Used as library source by address.C

CLING_EXPORT int gLibGlobal = 14;

#pragma pack(1)
CLING_EXPORT const char gByteAlign0 = 0, gByteAlign1 = 1, gByteAlign2 = 2, gByteAlign3 = 3;
