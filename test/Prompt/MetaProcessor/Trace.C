//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s
int abcdefghxyz = 10
//CHECK: (int) 10
.trace ast abcdefghxyz
//CHECK: Dumping abcdefghxyz:
//CHECK: VarDecl {{0x[0-9a-f]+}} <input_line_[[@LINE-3]]:2:2, col:20> col:6 used abcdefghxyz 'int' cinit
//CHECK: `-IntegerLiteral {{0x[0-9a-f]+}} <col:20> 'int' 10
