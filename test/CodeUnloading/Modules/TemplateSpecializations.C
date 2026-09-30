//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: rm -rf %t && mkdir -p %t
// RUN: cat %s | %cling -Xclang -fmodules -fimplicit-modules -Xclang -fmodules-cache-path=%t -fmodule-map-file=%S/Inputs/module.modulemap -I%S/Inputs 2>&1 | FileCheck %s

// Unloading a redeclaration of a class template visits all loaded
// specializations of the template. Dereferencing a specialization completes its
// redeclaration chain, if modules were loaded since, which lazily loads further
// specializations with the same template-argument hash from these modules into
// the specialization set that is being iterated over.

extern "C" int printf(const char*, ...);

#include "Tmpl.h"
// More than 8 loaded specializations, such that the set's vector lives on the
// heap and is reallocated when growing.
Tmpl<a::X>* x;
Tmpl<a::Y0>* v0; Tmpl<a::Y1>* v1; Tmpl<a::Y2>* v2; Tmpl<a::Y3>* v3;
Tmpl<a::Y4>* v4; Tmpl<a::Y5>* v5; Tmpl<a::Y6>* v6; Tmpl<a::Y7>* v7;
Tmpl<a::Y8>* v8; Tmpl<a::Y9>* v9;

// Loading another module makes the redeclaration chains of the specializations
// above out of date.
#include "TmplColliding.h"

template <class T> struct Tmpl;
.undo

printf("Unloaded\n"); // CHECK: Unloaded
Tmpl<b_42::X>* b42 = nullptr;
printf("%d\n", b42 == nullptr); // CHECK-NEXT: 1
.q
