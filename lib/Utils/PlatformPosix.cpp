//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author: Roman Zulak
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/Platform.h"

#if defined(LLVM_ON_UNIX)

#include <string>
#include <unistd.h>

// PATH_MAX
#ifdef __APPLE__
 #include <sys/syslimits.h>
#else
 #include <limits.h>
#endif

namespace cling {
namespace utils {
namespace platform {

std::string GetCwd() {
  char Buffer[PATH_MAX+1];
  if (::getcwd(Buffer, sizeof(Buffer)))
    return Buffer;

  ::perror("Could not get current working directory");
  return std::string();
}

} // namespace platform
} // namespace utils
} // namespace cling

#endif // LLVM_ON_UNIX
