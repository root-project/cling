//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

//RUN: cat %s | %cling -Xclang -verify 2>&1
// Test unknownTypeTest

typedef float vec4f __attribute__((ext_vector_type(4)));

vec4f testVar;
testVar // expected-error {{float __attribute__((ext_vector_type(4))) has unknown type, which is not supported for this kind of declaration}}

.q
