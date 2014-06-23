//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 -I %S -Xclang -verify
// Test enumTest

.rawInput 1
enum __attribute__((annotate("Enum.h"))) class EC;
.rawInput 0
EC x=EC::A; 
// expected-error {{}}

enum E:unsigned int;
template <typename T> class __attribute__((annotate("Def.h"))) Gen;
template <> class __attribute__((annotate("Enum.h"))) Gen<E>;

.rawInput 1
#include "Enum.h"
.rawInput 0
EC x=EC::A;

.q