//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Axel Naumann <axel@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#ifndef CLING_VISIBILITY_H
#define CLING_VISIBILITY_H

#include <new>

// Adapted from llvm/Support/Compiler.h
#ifndef __has_attribute
# define __has_attribute(x) 0
#endif

#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
// Adapted from https://gcc.gnu.org/wiki/Visibility
# ifdef __GNUC__
#  define CLING_LIB_EXPORT __attribute__ ((dllexport))
# else
#  define CLING_LIB_EXPORT __declspec(dllexport)
# endif
#elif (__has_attribute(visibility)) || defined(__GNUC__)
# define CLING_LIB_EXPORT __attribute__ ((visibility("default")))
#else
# define CLING_LIB_EXPORT
#endif

#endif // CLING_VISIBILITY_H
