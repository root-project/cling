//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_CLINGOPTIONS_H
#define CLING_CLINGOPTIONS_H

#include "llvm/Option/OptTable.h"

namespace cling {
namespace driver {
namespace clingoptions {
   enum ID {
    OPT_INVALID = 0, // This is not an option ID.
#define PREFIX(NAME, VALUE)
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "cling/Interpreter/ClingOptions.inc"
    LastOption
#undef OPTION
#undef PREFIX
   };
}
}
}
#endif // CLING_CLINGOPTIONS_H


