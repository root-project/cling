//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.

//------------------------------------------------------------------------------

// RUN: cat %s | %cling | FileCheck %s

#include <utility>
#include <memory>

int *i_ptr = nullptr
//CHECK: (int *) nullptr

std::unique_ptr<int> i_uptr
//CHECK: (std::unique_ptr<int> &) std::unique_ptr -> nullptr 

std::shared_ptr<int> i_sptr
//CHECK: (std::shared_ptr<int> &) std::shared_ptr -> nullptr

std::weak_ptr<int> i_wptr
//CHECK: (std::weak_ptr<int> &) std::weak_ptr -> nullptr

i_uptr = std::unique_ptr<int>(new int (3))
//CHECK: (std::unique_ptr &) std::unique_ptr -> 0x{{[0-9a-f]+}} 

i_uptr
//CHECK: (std::unique_ptr<int> &) std::unique_ptr -> 0x{{[0-9a-f]+}} 

i_sptr = std::make_shared<int>(6)
//CHECK: (std::shared_ptr &) std::shared_ptr -> 0x{{[0-9a-f]+}} 

i_sptr
//CHECK: (std::shared_ptr<int> &) std::shared_ptr -> 0x{{[0-9a-f]+}} 

i_wptr = i_sptr;
i_wptr
//CHECK: (std::weak_ptr<int> &) std::weak_ptr -> 0x{{[0-9a-f]+}}



