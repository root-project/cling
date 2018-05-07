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
// RUN: cat %s | %cling -x cuda -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <iostream>

// It compares the cuda module ctor and dtor of two random modules.
std::string ctor1, ctor2, dtor1, dtor2;

auto M = gCling->getLatestTransaction()->getModule();

for(auto I = M->begin(), E = M->end(); I != E; ++I){
  if((*I).getName().startswith_lower("__cuda_module_ctor_")){
    ctor1 = (*I).getName().str();
  }

  if((*I).getName().startswith_lower("__cuda_module_dtor_")){
    dtor1 = (*I).getName().str();
  }
}

// The last module should be another, because of the for loop.
M = gCling->getLatestTransaction()->getModule();

for(auto I = M->begin(), E = M->end(); I != E; ++I){
  if((*I).getName().startswith_lower("__cuda_module_ctor_")){
    ctor2 = (*I).getName().str();
  }

  if((*I).getName().startswith_lower("__cuda_module_dtor_")){
    dtor2 = (*I).getName().str();
  }
}

// Check if the ctor and dtor of the two modules are different.
bool result1 = ctor1 != ctor2
// CHECK: (bool) true
bool result2 = dtor1 != dtor2
// CHECK: (bool) true

// Check if the symbols are preprocessor compliant.
ctor1.find("__cuda_module_ctor_cling_module_")
// CHECK: (unsigned long) 0
dtor1.find("__cuda_module_dtor_cling_module_")
// CHECK: (unsigned long) 0

// expected-no-diagnostics
.q
