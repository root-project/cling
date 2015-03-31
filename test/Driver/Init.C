//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

//RUN: cat %s | %cling 2>&1 | FileCheck %s

#include <initializer_list>
auto l {'a', 'b', '\''};
l // CHECK: (class std::initializer_list<char> &) @0x{{.*}}
