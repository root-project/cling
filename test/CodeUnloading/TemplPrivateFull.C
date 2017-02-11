//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %built_cling --noruntime -Wno-undefined-inline -Xclang -verify 2>&1
// Test clangDeclInclude

#include <clang/AST/Decl.h>
.undo
// FIXME: Fails because of a lot of built-ins
// .compareState "Initial"

fabs // expected-error {{use of undeclared identifier 'fabs'}}

// Include it again though!
#include <clang/AST/Decl.h>
.undo

.q
