//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.

//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

.rawInput 1
// When begin() != end() but *begin() points to the container itself
// (in nlohmann::json), printValue_impl infinitely recurses without the
// self-reference check.

class RecursionTest {
public:
    RecursionTest() = default;
    auto begin() const { return this; } // iterate over self

    // Hard code the size of the collection to be exactly one and it 'just' contains this object.
    auto end() const { return this + 1; }
};
.rawInput 0

RecursionTest j;
j
// CHECK: (RecursionTest &) { <recursion detected> }

// expected-no-diagnostics
.q
