//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -DPROBABLY_DEFINED -DDEF_DEF -Xclang -verify 2>&1 | FileCheck %s
// Test preprocessorIfSupport

#if 0
  0
  // CHECK-NOT: (int) 0
#endif

#if 1
  1
  // CHECK: (int) 1
#endif

#if PROBABLY_NOT_DEFINED
  10
  // CHECK-NOT: (int) 10
#elif defined(PROBABLY_DEFINED)
  20
  // CHECK: (int) 20
#else
  30
  // CHECK-NOT: (int) 30
#endif//
// '#endif//' intentional

#ifdef DEF_DEF
  struct Nested {
    int array[1];
    Nested() { array[0] = 200; }
    int f() { return array[0]; }
  };
  Nested n  // Print expression works as last before #endif
  // CHECK: (Nested &) @0x{{[0-9a-f]+}}
#else
   Nested n
   Nested n;
   Nested n
#endif   
// '#endif   ' intentional

n.f()
// CHECK: (int) 200

#ifdef DEF_DEF
  80 // expected-error {{expected ';' after expression}}
  n.f();
#endif

.rawInput
#if 0
  Err Here
#elif PROBABLY_NOT_DEFINED
  Err Here
#else
  Nested n1;
#endif
.rawInput

n1.f()
// CHECK: (int) 200

.q
