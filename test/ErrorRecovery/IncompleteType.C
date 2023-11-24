//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s

#include <type_traits>

struct A { int v; };
std::is_default_constructible_v<A>
// CHECK: (const bool) true

struct B;
std::is_default_constructible_v<B>
// CHECK: incomplete type 'B'
struct B { int v; };
std::is_default_constructible_v<B>
// CHECK: (const bool) true

template <typename T> struct C;
template <> struct C<int>;
std::is_default_constructible_v<C<int>>
// CHECK: incomplete type 'C<int>'
template <> struct C<int> { int v; };
std::is_default_constructible_v<C<int>>
// CHECK: (const bool) true

.q
