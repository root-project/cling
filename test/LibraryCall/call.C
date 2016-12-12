//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: clang -shared -DCLING_EXPORT=%dllexport %S/call_lib.c -o%T/libcall_lib2%shlibext
// RUN: cat %s | %cling -L%T | FileCheck %s

.L libcall_lib2
extern "C" int cling_testlibrary_function();
int i = cling_testlibrary_function();
extern "C" int printf(const char* fmt, ...);
printf("got i=%d\n", i); // CHECK: got i=66
.q
