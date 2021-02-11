//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks, if the symbols __cuda_module_ctor and __cuda_module_dtor are
// unique for every module. Attention, for a working test case, a cuda
// fatbinary is necessary.
// RUN: cat %s | %cling -x cuda --cuda-path=%cudapath %cudasmlevel -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <iostream>

// Compare the cuda module ctor and dtor of two random modules.
std::string ctor1, ctor2, dtor1, dtor2;

auto T1 = gCling->getLatestTransaction();
// only __global__ CUDA kernel definitions has the cuda module ctor and dtor
__global__ void g1() { int i = 1; }
auto M1 = T1->getNext()->getModule();

for(auto &I : *M1){
  // The trailing '_' identify the function name as modified name.
  if(I.getName().startswith_lower("__cuda_module_ctor_")){
    ctor1 = I.getName().str();
  }

  if(I.getName().startswith_lower("__cuda_module_dtor_")){
    dtor1 = I.getName().str();
  }
}

auto T2 = gCling->getLatestTransaction();
__global__ void g2() { int i = 2; }
auto M2 = T2->getNext()->getModule();

// The two modules should have different names, because of the for loop.
M1->getName().str() != M2->getName().str()
// CHECK: (bool) true

for(auto &I : *M2){
  if(I.getName().startswith_lower("__cuda_module_ctor_")){
    ctor2 = I.getName().str();
  }

  if(I.getName().startswith_lower("__cuda_module_dtor_")){
    dtor2 = I.getName().str();
  }
}

// Check if the ctor and dtor of the two modules are different.
ctor1 != ctor2 // expected-note {{use '|=' to turn this inequality comparison into an or-assignment}}
// CHECK: (bool) true
dtor1 != dtor2 // expected-note {{use '|=' to turn this inequality comparison into an or-assignment}}
// CHECK: (bool) true

// Check if the ctor symbol starts with the correct prefix.
std::string expectedCtorPrefix = "__cuda_module_ctor_cling_module_";
ctor1.compare(0, expectedCtorPrefix.length(), expectedCtorPrefix)
// CHECK: (int) 0

// Check if the dtor symbol starts with the correct prefix.
std::string expectedDtorPrefix = "__cuda_module_dtor_cling_module_";
dtor1.compare(0, expectedDtorPrefix.length(), expectedDtorPrefix)
// CHECK: (int) 0

.q
