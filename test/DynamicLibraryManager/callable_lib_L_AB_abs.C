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
// RUN: %clang %fPIC -shared -DCLING_EXPORT=%dllexport %S/call_lib_L_AB.c -o%t-dir/lib/libcall_lib_L_AB%shlibext -L %t-dir/rlib -l%t-dir/rlib/libcall_lib_A.so -l%t-dir/rlib/libcall_lib_B.so
// RUN: cat %s | %cling -L%t-dir/lib 2>&1 | FileCheck %s

// Test: Lookup and load library Lib_L_AB that depends on two libraries Lib_A
//       and Lib_B thru absolute file names. Lib_A and Lib_B are outside
//       library loading path.

extern "C" int cling_testlibrary_function();
cling_testlibrary_function()
// CHECK: {{.*}}libcall_lib_L_AB{{.*}}
.L libcall_lib_L_AB
cling_testlibrary_function()
// CHECK: (int) 357

// Test: Second level library symbol lookup. Lookup function in Lib_A and print
// result value. LibA is not in loading path and symbols are reachable only via
// Lib_AB.

extern "C" int cling_testlibrary_function_A();
cling_testlibrary_function_A()
// CHECK: (int) 170

.q
