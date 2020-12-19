//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: mkdir -p %t-dir/rlib
// RUN: mkdir -p %t-dir/lib
// RUN: mkdir -p %t-dir/lib3
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_A.c -o%t-dir/rlib/libcall_lib_A%shlibext
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib_B.c -o%t-dir/rlib/libcall_lib_B%shlibext
// RUN: %clang %fPIC -shared -Wl,--disable-new-dtags -Wl,-rpath,%t-dir/rlib -DCLING_EXPORT=%dllexport %S/call_lib_L_AB.c -o%t-dir/lib/libcall_lib_L_AB%shlibext -L %t-dir/rlib -lcall_lib_A -lcall_lib_B
// RUN: %clang %fPIC -shared -Wl,--disable-new-dtags -Wl,-rpath,%t-dir/lib -DCLING_EXPORT=%dllexport %S/call_lib_L3.c -o%t-dir/lib3/libcall_lib_L3%shlibext -L %t-dir/lib -lcall_lib_L_AB
// RUN: cat %s | LD_LIBRARY_PATH="%t-dir/lib3" %cling 2>&1 | FileCheck %s

// Test: Lookup and load library Lib_L3 that depends on Lib_L_AB that depends on
//       two libraries Lib_A and Lib_B thru RPATH. Lib_L_AB, Lib_A and Lib_B are
//       not in LD_LIBRARY_PATH and are reachable only thru RPATHs.
//       This test try to lookup libraries and function in Lib_L3.

extern "C" int cling_testlibrary_function3();
cling_testlibrary_function3()
// CHECK: {{.*}}libcall_lib_L3{{.*}}
.L libcall_lib_L3
cling_testlibrary_function3()
// CHECK: (int) 360

// Test: Lookup and load library Lib_L3 that depends on Lib_L_AB that depends on
//       two libraries Lib_A and Lib_B thru RPATH. Lib_L_AB, Lib_A and Lib_B are
//       not in LD_LIBRARY_PATH and are reachable only thru RPATHs.
//       This test try to lookup libraries and function in second level library
//       Lib_L_AB.

extern "C" int cling_testlibrary_function();
cling_testlibrary_function()
// CHECK: (int) 357

// Test: Lookup and load library Lib_L3 that depends on Lib_L_AB that depends on
//       two libraries Lib_A and Lib_B thru RPATH. Lib_L_AB, Lib_A and Lib_B are
//       not in LD_LIBRARY_PATH and are reachable only thru RPATHs.
//       This test try to lookup libraries and function in third level library
//       Lib_A.

extern "C" int cling_testlibrary_function_A();
cling_testlibrary_function_A()
// CHECK: (int) 170

.q
