//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: mkdir -p %T/subdir && clang -DCLING_EXPORT=%dllexport -shared %S/call_lib.c -o %T/subdir/libtest%shlibext
// RUN: %setenv ENVVAR_LIB="%T/subdir" && %setenv ENVVAR_INC="%S/subdir"
// RUN: cat %s | %cling -I %S -Xclang -verify 2>&1 | FileCheck %s

#pragma cling add_include_path("$ENVVAR_INC")
#include "Include_header.h"
include_test()
// CHECK: OK(int) 0

#pragma cling add_library_path("$ENVVAR_LIB")
#pragma cling load("libtest")

#pragma cling add_library_path("$NONEXISTINGVARNAME")
//expected-no-diagnostics
