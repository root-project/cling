//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: mkdir -p %t-dir/lib
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_BB.c -o%t-dir/lib/libcall_lib_B%shlibext
// RUN: cat %s | %cling -L%t-dir/lib 2>&1 | FileCheck %s

// Test: Lookup and load library Lib_B. Lookup function and use it to print
//       result value. This value is diffrerent from another Lib_B variants
//       because this library is used for test library loading order.

.L libcall_lib_B
extern "C" int cling_testlibrary_function_B();
cling_testlibrary_function_B()
// CHECK: (int) 48059

.q
