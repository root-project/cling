//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify
// XFAIL: powerpc64
//This file checks a dereference of an invalid memory address pointer..
int *p = (int*)0x1;
*p; // expected-warning {{invalid memory pointer passed to a callee:}}
