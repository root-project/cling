//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "clang/AST/Decl.h"

const cling::LookupHelper& lookup = gCling->getLookupHelper();
cling::LookupHelper::DiagSetting diags = cling::LookupHelper::WithDiagnostics;

template <typename T> struct S {
   T var[3];
};

template <typename T> struct A {
   static void* fun() { return (void*)typename T::x(); }
};

const clang::Decl* G = lookup.findScope("", diags);
.storeState "beforeLookup"
lookup.findClassTemplate("S<void>", diags)
lookup.findFunctionArgs(G, "A<int>::fun", "0", diags);
.compareState "beforeLookup"
