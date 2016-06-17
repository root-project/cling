//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

//RUN: cat %s | %cling -Xclang -verify 2>&1
// Test unknownTypeTest

 #include <x86intrin.h>

 __m128 testVar;
testVar // expected-error@2 {{__attribute__((__vector_size__(4 * sizeof(float)))) float has unknown type, which is not supported for this kind of declaration}}

.q
