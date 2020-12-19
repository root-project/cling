//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: mkdir -p %t-dir/rlib
// RUN: mkdir -p %t-dir/lib
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_A.c -o%t-dir/rlib/libcall_lib_A%shlibext
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_B.c -o%t-dir/rlib/libcall_lib_B%shlibext
// RUN: %clang %fPIC -shared -DCLING_EXPORT=%dllexport %S/call_lib_L_AB.c -o%t-dir/lib/libcall_lib_L_AB%shlibext -L %t-dir/rlib -lcall_lib_A -lcall_lib_B
// RUN: cat %s | LD_LIBRARY_PATH="%t-dir/lib:%t-dir/rlib" %cling 2>&1 | FileCheck %s

// Test: Lookup and load libraries Lib_A, Lib_B and Lib_L_AB that depends on two
//       libraries Lib_A and Lib_B. Lib_A and Lib_B are also in LD_LIBRARY_PATH.

.L libcall_lib_A
.L libcall_lib_B
.L libcall_lib_L_AB
extern "C" int cling_testlibrary_function();
cling_testlibrary_function()
// CHECK: (int) 357

.q
