//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author: Guilherme Amadio <amadio@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include <cstdlib>
#include <cstring>

#ifndef CLING_UTILS_UTILS_H
#define CLING_UTILS_UTILS_H

#if defined(_MSC_VER) && !defined(strcasecmp)
#define strcasecmp _stricmp
#endif

namespace cling {
  namespace utils {
    /** Convert @p value to boolean */
    static inline bool ConvertEnvValueToBool(const char* value) {
      const char* true_strs[] = {"1", "true", "on", "yes"};

      if (!value)
        return false;

      for (auto str : true_strs)
        if (strcasecmp(value, str) == 0)
          return true;

      return false;
    }
  } // namespace utils
} // namespace cling

#endif // CLING_UTILS_UTILS_H
