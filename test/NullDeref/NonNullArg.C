//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling --ptrcheck -Xclang -verify | FileCheck %s
// XFAIL: powerpc64
//This file checks a call instruction. The called function has arguments with nonnull attribute.
extern "C" int printf(const char* fmt, ...);
.rawInput 1
extern "C" int cannotCallWithNull(int* p = 0);
extern "C" int cannotCallWithNull(int* p) __attribute__((nonnull(1))); //expected-note@1{{declared 'nonnull' here}}
extern "C" int cannotCallWithNull(int* p);
extern "C" int cannotCallWithNull(int* p);
.rawInput 0

extern "C" int cannotCallWithNull(int* p) {
  if (!p) // expected-warning {{nonnull parameter 'p' will evaluate to 'true' on first encounter}}
    printf("Must not be called with p=0.\n");
  return 1;
}
int *q = 0;
cannotCallWithNull(q); // expected-warning {{null passed to a callee that requires a non-null argument}}
//CHECK-NOT: Must not be called with p=0.
cannotCallWithNull(new int(4))
//CHECK: (int) 1



.q
