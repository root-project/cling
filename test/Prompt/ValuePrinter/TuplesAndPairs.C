//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Danilo Piparo <danilo.piparo@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.

//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s

make_pair("s",10)
// CHECK: (std::pair<std::{{[a-Z_]+}}<char const (&)[2]>::type, std::{{[a-Z_]+}}<int>::type>) { "s", 10 }

make_pair(4L,'c')
//CHECK: (std::pair<std::{{[a-Z_]+}}<long>::type, std::{{[a-Z_]+}}<char>::type>) { 4, 'c' }

make_tuple(2)
//CHECK: (std::tuple<std::{{[a-Z_]+}}<int>::type>) { 2 }

make_tuple(1.2f)
//CHECK: (std::tuple<std::{{[a-Z_]+}}<float>::type>) { 1.20000f }

make_tuple(1,make_tuple(1, 'c'))
//CHECK: (std::tuple<std::{{[a-Z_]+}}<int>::type, std::{{[a-Z_]+}}<std::tuple<int, char> >::type>) { 1, { 1, 'c' } }

.q
