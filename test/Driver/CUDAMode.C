//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author: Simeon Ehrig <s.ehrig@hzdr.de>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test starts the cling with the arg -xcuda and checks if the cuda mode
// is enabled.
// RUN: cat %s | %cling -x cuda --cuda-path=%cudapath %cudasmlevel -Xclang -verify 2>&1 | FileCheck %s
// RUN: cat %s | %cling -x cuda --cuda-path=%cudapath %cudasmlevel -fsyntax-only -Xclang -verify 2>&1
// REQUIRES: cuda-runtime


.rawInput 1
__global__ void foo(){ int i = 3; }
.rawInput 0

cudaError error
// CHECK: (cudaError) (cudaSuccess) : (unsigned int) 0

// expected-no-diagnostics
.q
