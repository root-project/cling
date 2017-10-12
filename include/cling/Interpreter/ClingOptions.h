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

namespace cling {
namespace driver {
namespace clingoptions {
   enum ID {
    OPT_INVALID = 0, // This is not an option ID.
#define PREFIX(NAME, VALUE)
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM, \
               HELPTEXT, METAVAR, VALUES) OPT_##ID,
#include "cling/Interpreter/ClingOptions.inc"
    LastOption
#undef OPTION
#undef PREFIX
   };
}
}
}
#endif // CLING_CLINGOPTIONS_H


