//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s
"simple"
//CHECK: (const char [7]) "simple"
"It's me"
//CHECK: (const char [8]) "It's me"
"Luke, I'm your (father({}{["
//CHECK: const char [28]) "Luke, I'm your (father({}{["
("http://foo/bar/whatever")
//CHECK: (const char [24]) "http://foo/bar/whatever"
("http://foo.bar/whatever")
//CHECK: (const char [24]) "http://foo.bar/whatever"
.q
