//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: mkdir -p %t-dir/lib
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_L.c -o%t-dir/lib/libcall_lib_L%shlibext
// RUN: cat %s | %cling -L%t-dir/lib 2>&1 | FileCheck %s

// Test: Lookup and load library Lib_L. Lookup function and use it to print
//       result value.

.L libcall_lib_L
extern "C" int cling_testlibrary_function();
cling_testlibrary_function()
// CHECK: (int) 88

.q
