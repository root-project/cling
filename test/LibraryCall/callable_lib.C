//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: clang -shared %S/call_lib.c -olibcall_lib%shlibext
// RUN: cat %s | %cling | FileCheck %s

.L libcall_lib
extern "C" int cling_testlibrary_function();

cling_testlibrary_function()
// CHECK: (int) 66
