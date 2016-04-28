//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s
template<int n> struct F{
  enum {RET=F<n-1>::RET*n} ;
};
template<> struct F<0> {
  enum {RET = 1};
};
F<7>::RET
//CHECK: (F<7>::(anonymous)) (F<7>::RET) : ({{(unsigned )?}}int) 5040

.q
