//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: mkdir -p %t-dir/lib
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_A.c -o%t-dir/lib/libcall_lib_A%shlibext
// RUN: cat %s | %cling -L%t-dir/lib 2>&1 | FileCheck %s

// Test: Lookup and load library Lib_A. Lookup function and use it to print
//       result value.

.L libcall_lib_A
extern "C" int cling_testlibrary_function_A();
cling_testlibrary_function_A()
// CHECK: (int) 170

.q
