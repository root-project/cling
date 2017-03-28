//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -I %S 2>&1 | FileCheck -allow-empty %s
// Test friendUnload

void bestFriend();

.storeState "NF"
#include "Friend.h"
.undo
.compareState "NF"
//CHECK-NOT: Differences

#include "FriendNested.h"
.undo
.compareState "NF"
//CHECK-NOT: Differences

#include "FriendRecursive.h"
.undo
.compareState "NF"
//CHECK-NOT: Differences


// STL has many of these in memory & stdexcept
#include <memory>
.undo
#include <memory>

//expected-no-diagnostics
.q
