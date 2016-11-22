//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.

//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

#include <string>
#include <tuple>
#include <vector>
#include <map>

std::vector<bool> Bv(5,5)
// FIXME: Printing std::vector<bool> is still broken.
// But the line above at least tests cling doesn't crash because of it.
// BROKENCHECK: (std::vector<bool> &) { true, true, true, true, true }

class CustomThing {
};

namespace cling {
  std::string printValue(const CustomThing *ptr) {
    return "";
  }
}

std::vector<CustomThing> A, B(1);
cling::printValue(&A) == cling::printValue(&B)
// CHECK: (bool) false

std::tuple<> tA
// CHECK: (std::tuple<> &) {}

std::map<int, int> m
// CHECK: (std::map<int, int> &) {}

for (int i = 0; i < 5; ++i) m[i] = i+1;

m
// CHECK: (std::map<int, int> &) { 0 => 1, 1 => 2, 2 => 3, 3 => 4, 4 => 5 }


// expected-no-diagnostics
.q
