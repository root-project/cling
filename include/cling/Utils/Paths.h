//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_UTILS_PATHS_H
#define CLING_UTILS_PATHS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace cling {
  namespace utils {

    ///\brief Collect the constituant paths from a PATH string.
    /// /bin:/usr/bin:/usr/local/bin -> {/bin, /usr/bin, /usr/local/bin}
    ///
    /// All paths returned existed at the time of the call
    /// \param [in] PathStr - The PATH string to be split
    /// \param [out] Paths - All the paths in the string that exist
    /// \param [in] EarlyOut - If any path doesn't exist stop and return false
    /// \param [in] Delim - The delimeter to use
    ///
    /// \return true if all paths existed, otherwise false
    ///
    bool SplitPaths(llvm::StringRef PathStr,
                    llvm::SmallVectorImpl<llvm::StringRef>& Paths,
                    bool EarlyOut = false,
                    llvm::StringRef Delim = llvm::StringRef(":"));
  }
}

#endif // CLING_UTILS_PATHS_H
