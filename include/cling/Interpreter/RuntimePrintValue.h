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

namespace cling {

  class Value;

  // General fallback - prints the address
  std::string printValue(const void *ptr);

  // void pointer
  std::string printValue(const void **ptr);

  // Bool
  std::string printValue(const bool *val);

  // Chars
  std::string printValue(const char *val);

  std::string printValue(const signed char *val);

  std::string printValue(const unsigned char *val);

  // Ints
  std::string printValue(const short *val);

  std::string printValue(const unsigned short *val);

  std::string printValue(const int *val);

  std::string printValue(const unsigned int *val);

  std::string printValue(const long *val);

  std::string printValue(const unsigned long *val);

  std::string printValue(const long long *val);

  std::string printValue(const unsigned long long *val);

  // Reals
  std::string printValue(const float *val);

  std::string printValue(const double *val);

  std::string printValue(const long double *val);

  // Char pointers
  std::string printValue(const char *const *val);

  std::string printValue(const char **val);

  // std::string
  std::string printValue(const std::string *val);

  // cling::Value
  std::string printValue(const Value *value);

  // Collections internal declaration
  namespace collectionPrinterInternal {
    // Maps declaration
    template<typename CollectionType>
    auto printValue_impl(const CollectionType *obj, short)
      -> decltype(
      ++(obj->begin()), obj->end(),
        obj->begin()->first, obj->begin()->second,
        std::string());

    // Vector, set, deque etc. declaration
    template<typename CollectionType>
    auto printValue_impl(const CollectionType *obj, int)
      -> decltype(
      ++(obj->begin()), obj->end(),
        *(obj->begin()),
        std::string());

    // No general fallback anymore here, void* overload used for that now
  }

  // Collections
  template<typename CollectionType>
  auto printValue(const CollectionType *obj)
  -> decltype(collectionPrinterInternal::printValue_impl(obj, 0), std::string())
  {
    return collectionPrinterInternal::printValue_impl(obj, (short)0);  // short -> int -> long = priority order
  }

  // Arrays
  template<typename T, size_t N>
  std::string printValue(const T (*obj)[N]) {
    std::string str = "{ ";

    for (int i = 0; i < N; ++i) {
      str += printValue(*obj + i);
      if (i < N - 1) str += ", ";
    }

    return str + " }";
  }

  // Collections internal
  namespace collectionPrinterInternal {
    // Maps
    template<typename CollectionType>
    auto printValue_impl(const CollectionType *obj, short)
    -> decltype(
    ++(obj->begin()), obj->end(),
        obj->begin()->first, obj->begin()->second,
        std::string())
    {
      std::string str = "{ ";

      auto iter = obj->begin();
      auto iterEnd = obj->end();
      while (iter != iterEnd) {
        str += printValue(&iter->first);
        str += " => ";
        str += printValue(&iter->second);
        ++iter;
        if (iter != iterEnd) {
          str += ", ";
        }
      }

      return str + " }";
    }

    // Vector, set, deque etc.
    template<typename CollectionType>
    auto printValue_impl(const CollectionType *obj, int)
    -> decltype(
    ++(obj->begin()), obj->end(),
        *(obj->begin()),
        std::string())
    {
      std::string str = "{ ";

      auto iter = obj->begin();
      auto iterEnd = obj->end();
      while (iter != iterEnd) {
        str += printValue(&(*iter));
        ++iter;
        if (iter != iterEnd) {
          str += ", ";
        }
      }

      return str + " }";
    }
  }

}

#endif
