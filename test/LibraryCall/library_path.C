//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// REQUIRES: shell
// RUN: env -i LD_LIBRARY_PATH="../:./../" %cling ".L" | FileCheck %s

// CHECK: ../
// CHECK-NEXT: ./../
// CHECK-NEXT: .
// CHECK-NEXT: [system]
