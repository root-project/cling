//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Boris Perovic <boris.perovic@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------
#ifndef CLING_RUNTIME_PRINT_VALUE_H
#define CLING_RUNTIME_PRINT_VALUE_H

#include <string>
#include "Value.h"

#ifndef _Bool
#define _Bool bool
#endif

namespace cling {

  // void pointer
  std::string printValue(void *ptr);

  // Bool
  std::string printValue(bool val);

  // Chars
  std::string printValue(char val);

  std::string printValue(signed char val);

  std::string printValue(unsigned char val);

  // Ints
  std::string printValue(short val);

  std::string printValue(unsigned short val);

  std::string printValue(int val);

  std::string printValue(unsigned int val);

  std::string printValue(long val);

  std::string printValue(unsigned long val);

  std::string printValue(long long val);

  std::string printValue(unsigned long long val);

  // Reals
  std::string printValue(float val);

  std::string printValue(double val);

  std::string printValue(long double val);

  // Char pointers
  std::string printValue(const char *const val);

  std::string printValue(char *val);

  // std::string
  std::string printValue(const std::string &val);

  // cling::Value
  std::string printValue(const Value &value);

  // Maps declaration
  template<typename CollectionType>
  auto printValue_impl(const CollectionType &obj, short)
      -> decltype(
      ++(obj.begin()), obj.end(),
          obj.begin()->first, obj.begin()->second,
          std::string());

  // Collections like vector, set, deque etc. declaration
  template<typename CollectionType>
  auto printValue_impl(const CollectionType &obj, int)
      -> decltype(
      ++(obj.begin()), obj.end(),
          *(obj.begin()),
          std::string());

  // General fallback - print object address declaration
  template<typename T>
  std::string printValue_impl(const T &obj, long);

  // Collections and general fallback entry function
  template<typename CollectionType>
  auto printValue(const CollectionType &obj)
  -> decltype(printValue_impl(obj, 0), std::string())
  {
    return printValue_impl(obj, (short)0);  // short -> int -> long = priority order
  }

  // Arrays
  template<typename T, size_t N>
  std::string printValue(const T (&obj)[N]) {
    std::string str = "{ ";

    for(int i = 0; i < N; ++i) {
      str = str + printValue(obj[i]);
      if (i < N-1) str = str + ", ";
    }

    return str + " }";
  }

  // Maps
  template<typename CollectionType>
  auto printValue_impl(const CollectionType &obj, short)
      -> decltype(
          ++(obj.begin()), obj.end(),
          obj.begin()->first, obj.begin()->second,
          std::string())
  {
    std::string str = "{ ";

    auto iter = obj.begin();
    auto iterEnd = obj.end();
    while (iter != iterEnd) {
      str = str + printValue(iter->first);
      str = str + " => ";
      str = str + printValue(iter->second);
      ++iter;
      if (iter != iterEnd) str = str + ", ";
    }

    return str + " }";
  }

  // Collections like vector, set, deque etc.
  template<typename CollectionType>
  auto printValue_impl(const CollectionType &obj, int)
      -> decltype(
          ++(obj.begin()), obj.end(),
          *(obj.begin()),
          std::string())
  {
    std::string str = "{ ";

    auto iter = obj.begin();
    auto iterEnd = obj.end();
    while (iter != iterEnd) {
      str = str + printValue(*iter);
      ++iter;
      if (iter != iterEnd) str = str + ", ";
    }

    return str + " }";
  }

  // General fallback - print object address
  template<typename T>
  std::string printValue_impl(const T &obj, long) {
    return "@" + printValue((void *) &obj);
  }

}

#endif
