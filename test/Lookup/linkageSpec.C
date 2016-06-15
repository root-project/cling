//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling %s -Xclang -verify 2>&1 | FileCheck %s

// This test assures that extern "C" declarations are usable via .x

extern "C" int printf(const char*, ...);

extern "C" int linkageSpec() {
  printf("linkageSpec called %s\n", __FUNCTION__);
  return 101;
}

// CHECK: linkageSpec called linkageSpec
// CHECK: (int) 101
// expected-no-diagnostics
