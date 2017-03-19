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
#include <set>

std::vector<bool> Bv(5,5)
// CHECK: (std::vector<bool> &) { true, true, true, true, true }

std::vector<std::vector<bool>> Bvv;
for (int i = 0; i < 5; ++i) {
  Bvv.push_back(std::vector<bool>());
  for (int j = 0, N = i+1; j < N; ++j)
    Bvv.back().push_back(j % 4);
}
Bvv.push_back(std::vector<bool>());
Bvv
// CHECK-NEXT: (std::vector<std::vector<bool> > &) { { false }, { false, true }, { false, true, true }, { false, true, true, true }, { false, true, true, true, false }, {} }

class CustomThing {
};

namespace cling {
  std::string printValue(const CustomThing *ptr) {
    return "";
  }
}

std::vector<CustomThing> A, B(1);
cling::printValue(&A) == cling::printValue(&B)
// CHECK-NEXT: (bool) false

std::tuple<> tA
// CHECK-NEXT: (std::tuple<> &) {}

std::map<int, int> M
// CHECK-NEXT: (std::map<int, int> &) {}
std::map<int, std::pair<int,int> > M2;
std::map<std::pair<std::string,bool>, std::pair<int,bool> > M3;
std::set<std::pair<int, std::string>> S;
std::map<std::string, std::map<int, std::pair<int,std::string>>> MM;

for (int i = 0; i < 5; ++i) {
  const std::string Str = std::to_string(i);
  M[i] = i+1;
  M2[i] = std::make_pair(i+1, i+2);
  M3[std::make_pair(Str, i%2)] = std::make_pair(i*10, i%3);
  S.insert(std::make_pair(i*10+4, Str));

  auto &MMv = MM[Str];
  for (int j = 0; j < 3; ++j) {
    MMv[j] = std::make_pair(j*3, Str + std::to_string(j*5));
  }
}

M
// CHECK-NEXT: (std::map<int, int> &) { 0 => 1, 1 => 2, 2 => 3, 3 => 4, 4 => 5 }

M2
// CHECK-NEXT: (std::map<int, std::pair<int, int> > &) { 0 => {1 , 2}, 1 => {2 , 3}, 2 => {3 , 4}, 3 => {4 , 5}, 4 => {5 , 6} }

M3
// CHECK-NEXT: (std::map<std::pair<std::string, bool>, std::pair<int, bool> > &) { {"0" , false} => {0 , false}, {"1" , true} => {10 , true}, {"2" , false} => {20 , true}, {"3" , true} => {30 , false}, {"4" , false} => {40 , true} }

S
// CHECK-NEXT: (std::set<std::pair<int, std::string> > &) { {4 , "0"}, {14 , "1"}, {24 , "2"}, {34 , "3"}, {44 , "4"} }

MM
// (std::map<std::string, std::map<int, std::pair<int, std::string> > > &) { "0" => { 0 => {0 , "00"}, 1 => {3 , "05"}, 2 => {6 , "010"} }, "1" => { 0 => {0 , "10"}, 1 => {3 , "15"}, 2 => {6 , "110"} }, "2" => { 0 => {0 , "20"}, 1 => {3 , "25"}, 2 => {6 , "210"} }, "3" => { 0 => {0 , "30"}, 1 => {3 , "35"}, 2 => {6 , "310"} }, "4" => { 0 => {0 , "40"}, 1 => {3 , "45"}, 2 => {6 , "410"} } }

// expected-no-diagnostics
.q
