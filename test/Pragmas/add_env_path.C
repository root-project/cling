//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: mkdir -p %T/subdir && clang -DCLING_EXPORT=%dllexport -shared %S/call_lib.c -o %T/subdir/libtest%shlibext
// RUN: cat %s | %cling -I %S -DENVVAR_LIB="\"%/T/subdir\"" -DENVVAR_INC="\"%/p/subdir\"" -Xclang -verify 2>&1 | FileCheck %s

extern "C" int cling_testlibrary_function();

// For gcc setenv
#ifndef __THROW
 #define __THROW
#endif
extern "C" int setenv(const char *name, const char *value, int overwrite) __THROW;
extern "C" int _putenv_s(const char *name, const char *value);
static void setup() {
#ifdef _WIN32
 #define setenv(n, v, o) _putenv_s(n,v)
#endif
  ::setenv("ENVVAR_INC", ENVVAR_INC, 1);
  ::setenv("ENVVAR_LIB", ENVVAR_LIB, 1);
}
setup();

#pragma cling add_include_path("$ENVVAR_INC")
#include "Include_header.h"
include_test()
// CHECK: OK(int) 0

#pragma cling add_library_path("$ENVVAR_LIB")
#pragma cling load("libtest")
cling_testlibrary_function()
// CHECK: (int) 66

#pragma cling add_library_path("$NONEXISTINGVARNAME")
//expected-no-diagnostics
