//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I%p | FileCheck %s

// This file should be used as regression test for the meta processing subsystem
// Reproducers of fixed bugs should be put here

// PR #93092
// Don't remove the spaces and tabs
.L     cling/Interpreter/Interpreter.h    
.X ./DotXable.h(5)
// CHECK: 5
// End PR #93092
