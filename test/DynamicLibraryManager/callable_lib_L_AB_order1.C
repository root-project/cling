//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: mkdir -p %t-dir/rlib1
// RUN: mkdir -p %t-dir/rlib2
// RUN: mkdir -p %t-dir/lib
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_A.c -o%t-dir/rlib1/libcall_lib_A%shlibext
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_B.c -o%t-dir/rlib1/libcall_lib_B%shlibext
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_AA.c -o%t-dir/rlib2/libcall_lib_A%shlibext
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_BB.c -o%t-dir/rlib2/libcall_lib_B%shlibext
// RUN: %clang %fPIC -shared -Wl,-rpath,%t-dir/rlib1 -DCLING_EXPORT=%dllexport %S/call_lib_L_AB.c -o%t-dir/lib/libcall_lib_L_AB%shlibext -L %t-dir/rlib1 -lcall_lib_A -lcall_lib_B
// RUN: cat %s | LD_LIBRARY_PATH="%t-dir/lib:%t-dir/rlib2" %cling 2>&1 | FileCheck %s

// Test: Lookup and load library Lib_L_AB that depends on two libraries Lib_A
//       and Lib_B via RUNPATH. Lib_A and Lib_B are also in LD_LIBRARY_PATH.
//       Lib_L_AB must load and use libraries in LD_LIBRARY_PATH first.

extern "C" int cling_testlibrary_function();
cling_testlibrary_function()
// CHECK: {{.*}}libcall_lib_L_AB{{.*}}
.L libcall_lib_L_AB
cling_testlibrary_function()
// CHECK: (int) 91749
357

.q
