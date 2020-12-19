//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: mkdir -p %t-dir/lib
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib.c -o%t-dir/lib/libcall_lib%shlibext
// RUN: cat %s | %cling -L %t-dir/lib 2>&1 | FileCheck %s

// Test: Cling pragma for loading libraries. Cling shows error when library
//       do not exists (do not reacheble via library PATH.

#pragma cling load("DoesNotExistPleaseRecover")
// expected-error@input_line_13:1{{'DoesNotExistPleaseRecover' file not found}}

// Test: Cling pragma for loading libraries. Lookup and load library call_lib.
//       Lookup functions from libraries and use them to print result value.

#pragma cling load("libcall_lib")
extern "C" int cling_testlibrary_function();
cling_testlibrary_function()
// CHECK: (int) 42

.q
