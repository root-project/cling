//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S -Xclang -verify 2>&1 | FileCheck %s
// Test inlineNamespaces

#define TEST_NAMESPACE test
#include "NameSpaces.h"
.undo
#include "NameSpaces.h"
.undo
#undef TEST_NAMESPACE

#define TEST_NAMESPACE std
#include "NameSpaces.h"
.undo
#include "NameSpaces.h"
.undo
#undef TEST_NAMESPACE


#include <stdexcept>
.undo
#include <stdexcept>
.undo

101
// CHECK: (int) 101

// expected-no-diagnostics
.q
