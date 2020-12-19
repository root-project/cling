//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: mkdir -p %t-dir/lib
// RUN: %clang -shared -DCLING_EXPORT=%dllexport %S/call_lib.c -o%t-dir/lib/libcall_lib%shlibext
// RUN: cat %s | %cling -L%t-dir/lib 2>&1 | FileCheck %s

// Test: Lookup and load library call_lib. Lookup functions from libraries and
//       use them to print result value.

.L libcall_lib
extern "C" int cling_testlibrary_function();
int i = cling_testlibrary_function();
extern "C" int printf(const char* fmt, ...);
printf("got i=%d\n", i); // CHECK: got i=42

.q
