//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling %s -Xclang -verify 2>&1 | FileCheck %s

// Check that explicit instantiation of a template after an implicit
// instantiation of a constexpr member does not trigger an assertion.

extern "C" int printf(const char* fmt, ...);

template <class T> struct Box {
  constexpr Box() : {}
};

Box<int> b; // Trigger implicit instantiation
template class Box<int>; // Explicit instantiation, forces re-instantiation

printf("Ran without assertions\n"); // CHECK: Ran without assertions

// expected-no-diagnostics
