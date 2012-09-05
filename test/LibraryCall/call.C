// RUN: clang -shared %S/call_lib.c -olibcall_lib%shlibext
// RUN: cat %s | %cling | FileCheck %s

.L libcall_lib
extern "C" int cling_testlibrary_function();
int i = cling_testlibrary_function();
extern "C" int printf(const char* fmt, ...);
printf("got i=%d\n", i); // CHECK: got i=66
.q
