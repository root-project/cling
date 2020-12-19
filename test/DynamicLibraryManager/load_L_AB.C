//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: mkdir -p %t-dir/lib
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_A.c -o%t-dir/lib/libcall_lib_A%shlibext
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_B.c -o%t-dir/lib/libcall_lib_B%shlibext
// RUN: %clang %fPIC -shared -DCLING_EXPORT=%dllexport %S/call_lib_L_AB.c -o%t-dir/lib/libcall_lib_L_AB%shlibext -L %t-dir/lib -lcall_lib_A -lcall_lib_B
// RUN: cat %s | LD_LIBRARY_PATH="%t-dir/lib" %cling 2>&1 | FileCheck %s

// Test: Cling pragma for loading libraries. Lookup and load library Lib_L_AB
//       that depends on two libraries Lib_A and Lib_B via file names.
//       Lib_L_AB, Lib_A and Lib_B are in LD_LIBRARY_PATH.
//       Lookup functions from libraries and use them to print result value.

#pragma cling load("libcall_lib_L_AB")
extern "C" int cling_testlibrary_function();
cling_testlibrary_function()
// CHECK: (int) 357

.q
