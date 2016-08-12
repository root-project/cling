//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Paths.h"
#include "llvm/Support/FileSystem.h"

namespace cling {
namespace utils {

bool SplitPaths(llvm::StringRef PathStr,
                llvm::SmallVectorImpl<llvm::StringRef>& Paths, bool EarlyOut,
                llvm::StringRef Delim) {
  bool AllExisted = true;
  for (std::pair<llvm::StringRef, llvm::StringRef> Split = PathStr.split(Delim);
       !Split.second.empty(); Split = PathStr.split(Delim)) {
    if (!llvm::sys::fs::is_directory(Split.first)) {
      if (EarlyOut)
        return false;
      AllExisted = false;
    } else
      Paths.push_back(Split.first);
    PathStr = Split.second;
  }

  // Add remaining part
  if (llvm::sys::fs::is_directory(PathStr))
    Paths.push_back(PathStr);
  else
    AllExisted = false;

  return AllExisted;
}

} // namespace utils
} // namespace cling
