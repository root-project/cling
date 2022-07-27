//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// The Test checks, if the symbols __cuda_module_ctor, __cuda_module_dtor,
// __cuda_register_globals, __cuda_fatbin_wrapper and __cuda_gpubin_handle are
// unique for every module. Attention, for a working test case, a cuda fatbinary
// is necessary.
// RUN: cat %s | %cling -x cuda --cuda-path=%cudapath %cudasmlevel -Xclang -verify 2>&1 | FileCheck %s
// REQUIRES: cuda-runtime

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <iostream>
#include <map>
#include <vector>

// for each key in the map, the test searches for a symbol in a module and
// stores the full name in the vector in the value
std::map<std::string, std::vector<std::string>> module_compare;
for (std::string const& s :
     {"__cuda_module_ctor", "__cuda_module_dtor", "__cuda_register_globals",
      "__cuda_fatbin_wrapper", "__cuda_gpubin_handle"}) {
  module_compare.emplace(s, std::vector<std::string>{});
}

auto T1 = gCling->getLatestTransaction();
// only __global__ CUDA kernel definitions has the cuda specific functions and
// variables to register a kernel
__global__ void g1() { int i = 1; }
auto M1 = T1->getNext()->getCompiledModule();

// search for the symbols in the llvm::Module M1
for (auto const& key_value : module_compare) {
  for (auto& I : *M1) {
    if (I.getName().startswith(key_value.first)) {
      module_compare[key_value.first].push_back(I.getName().str());
    }
  }
  for (auto& I : M1->globals()) {
    if (I.getName().startswith(key_value.first)) {
      module_compare[key_value.first].push_back(I.getName().str());
    }
  }
}

// verify, that each symbol was found in the module
// if a symbol was not found, the vector should be empty
for (auto const& key_value : module_compare) {
  if (key_value.second.size() < 1) {
    std::cout << "could not find symbol" << std::endl;
    // CHECK-NOT: could not find symbol
    std::cout << "\"" << key_value.first << "\" is not in transaction T1"
              << std::endl;
  }
}

auto T2 = gCling->getLatestTransaction();
__global__ void g2() { int i = 2; }
auto M2 = T2->getNext()->getCompiledModule();

// search for the symbols in the llvm::Module M2
for (auto const& key_value : module_compare) {
  for (auto& I : *M2) {
    if (I.getName().startswith(key_value.first)) {
      module_compare[key_value.first].push_back(I.getName().str());
    }
  }
  for (auto& I : M2->globals()) {
    if (I.getName().startswith(key_value.first)) {
      module_compare[key_value.first].push_back(I.getName().str());
    }
  }
}

// verify, that each symbol was found in the second module
for (auto const& key_value : module_compare) {
  if (key_value.second.size() < 2) {
    std::cout << "could not find symbol" << std::endl;
    // CHECK-NOT: could not find symbol
    std::cout << "\"" << key_value.first << "\" is not in transaction T2"
              << std::endl;
  }
}

for (auto const& key_value : module_compare) {
  std::string const generic_symbol_name = key_value.first;
  std::string const symbol_name_suffix = generic_symbol_name + "_cling_module_";
  std::string const T1_symbol_name = key_value.second[0];
  std::string const T2_symbol_name = key_value.second[1];

  // check if each symbols are different for different modules
  if (T1_symbol_name != T2_symbol_name) {
    std::cout << "T1_symbol_name and T2_symbol_name are unique" << std::endl;
    // CHECK: T1_symbol_name and T2_symbol_name are unique
  } else {
    std::cerr << "T1_symbol_name and T2_symbol_name are equals" << std::endl;
    // CHECK-NOT: T1_symbol_name and T2_symbol_name are equals
    std::cerr << T1_symbol_name << " == " << T2_symbol_name << std::endl;
  }

  // only the module number is difference for each symbol
  // therefor the begin of the symbol name can be checked
  if (0 != T1_symbol_name.compare(0, symbol_name_suffix.length(),
                                  symbol_name_suffix)) {
    std::cerr << "Wrong suffix" << std::endl;
    // CHECK-NOT: Wrong suffix
    std::cerr << "T1_symbol_name: " << T1_symbol_name << std::endl;
    std::cerr << "expected symbol + suffix: " << symbol_name_suffix
              << std::endl;
  }

  if (0 != T2_symbol_name.compare(0, symbol_name_suffix.length(),
                                  symbol_name_suffix)) {
    std::cerr << "Wrong suffix" << std::endl;
    // CHECK-NOT: Wrong suffix
    std::cerr << "T2_symbol_name: " << T2_symbol_name << std::endl;
    std::cerr << "expected symbol + suffix: " << symbol_name_suffix
              << std::endl;
  }
}

// expected-no-diagnostics
.q
