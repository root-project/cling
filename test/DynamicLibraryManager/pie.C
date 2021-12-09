//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// REQUIRES: not_system-windows

// RUN: mkdir -p %t-dir/lib
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_A.c -o%t-dir/lib/libcall_lib_A%shlibext
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_B.c -o%t-dir/lib/libcall_lib_B%shlibext
// RUN: %clang %fPIC -fpie -pie -DCLING_EXPORT=%dllexport %S/call_pie.c -o%t-dir/lib/call_pie_so%shlibext
// RUN: %clang %fPIC -fpie -pie -DCLING_EXPORT=%dllexport %S/call_pie.c -o%t-dir/lib/call_pie
// RUN: cat %s | LD_LIBRARY_PATH="%t-dir/lib" %cling 2>&1 | FileCheck %s

// Test: Cling pragma for loading libraries. Lookup and load library Lib_L_AB
//       that depends on two libraries Lib_A and Lib_B via file names.
//       Lib_A and Lib_B are in LD_LIBRARY_PATH. Call_pie is "PIE" compiled excutable.
//       Lookup functions from libraries and use them to print result value.
//       We are expecting an error for the symbol from the PIE executable, which is
//       expected not to be loaded by dynamic linker.

#pragma cling load("libcall_lib_A")
extern "C" int cling_testlibrary_function_A();
cling_testlibrary_function_A()
// CHECK: (int) 170

#pragma cling load("libcall_lib_B")
extern "C" int cling_testlibrary_function_B();
cling_testlibrary_function_B()
// CHECK: (int) 187

extern "C" int cling_testlibrary_function();
cling_testlibrary_function()
// expected-error {{symbol 'cling_testlibrary_function' unresolved while linking}}

//.q
