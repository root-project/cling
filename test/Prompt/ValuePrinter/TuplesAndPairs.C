//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Danilo Piparo <danilo.piparo@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.

//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s

#include <utility>
#include <tuple>

std::make_pair("s",10)
//CHECK: (std::pair<{{.+char.+\[2\].*,.*int.*}}>) { "s", 10 }

std::make_pair(4L,'c')
//CHECK: (std::pair<{{.*long.*,.*char.*}}>) { 4, 'c' }

std::make_tuple()
//CHECK: (std::tuple<>) {}

std::make_tuple(2)
//CHECK: (std::tuple<{{.*int.*}}>) { 2 }

std::make_tuple(1.2f)
//CHECK: (std::tuple<{{.*float.*}}>) { 1.20000f }

std::make_tuple(1, std::make_tuple(1, 'c'))
//CHECK: (std::tuple<{{.*int.*,.*std::tuple<.*int,.*char.*>.*}}>) { 1, { 1, 'c' } }

.q
