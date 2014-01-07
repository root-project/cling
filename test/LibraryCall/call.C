//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: clang -shared %S/call_lib.c -o%p/libcall_lib%shlibext && ls %p/libcall_lib%shlibext && cat %s | %cling -L%p | FileCheck %s

.L libcall_lib
extern "C" int cling_testlibrary_function();
int i = cling_testlibrary_function();
extern "C" int printf(const char* fmt, ...);
printf("got i=%d\n", i); // CHECK: got i=66
.q
