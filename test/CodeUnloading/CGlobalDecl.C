//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%S -Xclang -verify  | FileCheck %s
// Test externC_GlobalDecl

extern "C" int printf(const char*,...);

printf("Starting C++ globals check\n");
// CHECK: Starting C++ globals check

#define BEGIN_DECL_
#define _END_DECL
#include "CGlobalDecl.h"
.undo
#include "CGlobalDecl.h"
.undo

printf("C++ globals included and undone\n");
// CHECK: C++ globals included and undone

printf("Starting C globals check\n");
// CHECK: Starting C globals check

#undef BEGIN_DECL_
#undef _END_DECL

#define BEGIN_DECL_  extern "C" {
#define _END_DECL    }
#include "CGlobalDecl.h"
.undo
#include "CGlobalDecl.h"

printf("C globals included and undone\n");
// CHECK: C globals included and undone

// expected-no-diagnostics
.q
