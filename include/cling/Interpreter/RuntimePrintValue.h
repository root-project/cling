//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Boris Perovic <boris.perovic@cern.ch>
// author:  Danilo Piparo <danilo.piparo@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------
#ifndef CLING_RUNTIME_PRINT_VALUE_H
#define CLING_RUNTIME_PRINT_VALUE_H

#if !defined(__CLING__)
#error "This file must not be included by compiled programs."
#endif

#include <cling/Interpreter/Visibility.h>

#if __cplusplus >= 201703L
#include <filesystem>
#endif
#include <memory>
#if __cplusplus >= 202002L
#include <version>
#endif
#ifdef __cpp_lib_source_location
#include <source_location>
#endif
#include <string>
#include <tuple>
#include <type_traits>

namespace cling {

  class Value;

  // General fallback - prints the address
  CLING_LIB_EXPORT
  std::string printValue(const void *ptr);

  // Fallback for e.g. vector<bool>'s bit iterator:
  template <class T,
    class = typename std::enable_if<!std::is_pointer<T>::value>::type>
  inline std::string printValue(const T& val) { return "{not representable}"; }

  // void pointer
  CLING_LIB_EXPORT
  std::string printValue(const void **ptr);

  // Bool
  CLING_LIB_EXPORT
  std::string printValue(const bool *val);

  // Chars
  CLING_LIB_EXPORT
  std::string printValue(const char *val);

  CLING_LIB_EXPORT
  std::string printValue(const signed char *val);

  CLING_LIB_EXPORT
  std::string printValue(const unsigned char *val);

  CLING_LIB_EXPORT
  std::string printValue(const char16_t *val);

  CLING_LIB_EXPORT
  std::string printValue(const char32_t *val);

  CLING_LIB_EXPORT
  std::string printValue(const wchar_t *val);

  // Ints
  CLING_LIB_EXPORT
  std::string printValue(const short *val);

  CLING_LIB_EXPORT
  std::string printValue(const unsigned short *val);

  CLING_LIB_EXPORT
  std::string printValue(const int *val);

  CLING_LIB_EXPORT
  std::string printValue(const unsigned int *val);

  CLING_LIB_EXPORT
  std::string printValue(const long *val);

  CLING_LIB_EXPORT
  std::string printValue(const unsigned long *val);

  CLING_LIB_EXPORT
  std::string printValue(const long long *val);

  CLING_LIB_EXPORT
  std::string printValue(const unsigned long long *val);

  // Reals
  CLING_LIB_EXPORT
  std::string printValue(const float *val);

  CLING_LIB_EXPORT
  std::string printValue(const double *val);

  CLING_LIB_EXPORT
  std::string printValue(const long double *val);

  // Char pointers
  CLING_LIB_EXPORT
  std::string printValue(const char *const *val);

  CLING_LIB_EXPORT
  std::string printValue(const char **val);

  // std::string
  CLING_LIB_EXPORT
  std::string printValue(const std::string *val);

  CLING_LIB_EXPORT
  std::string printValue(const std::wstring *val);

  CLING_LIB_EXPORT
  std::string printValue(const std::u16string *val);

  CLING_LIB_EXPORT
  std::string printValue(const std::u32string *val);

  // constant unicode strings, i.e. u"String"
  template <typename T>
  CLING_LIB_EXPORT
  std::string toUTF8(const T* const Src, size_t N, const char Prefix = 0);

  template <size_t N>
  inline std::string printValue(char16_t const (*val)[N]) {
    return toUTF8(reinterpret_cast<const char16_t * const>(val), N, 'u');
  }

  template <size_t N>
  inline std::string printValue(char32_t const (*val)[N]) {
    return toUTF8(reinterpret_cast<const char32_t * const>(val), N, 'U');
  }

  template <size_t N>
  inline std::string printValue(wchar_t const (*val)[N]) {
    return toUTF8(reinterpret_cast<const wchar_t * const>(val), N, 'L');
  }

  template <size_t N>
  inline std::string printValue(char const (*val)[N]) {
    return toUTF8(reinterpret_cast<const char * const>(val), N, 1);
  }

#ifdef __cpp_lib_source_location
  CLING_LIB_EXPORT
  std::string printValue(const std::source_location* location);
#endif

  // cling::Value
  CLING_LIB_EXPORT
  std::string printValue(const Value *value);

  namespace valuePrinterInternal {
    CLING_LIB_EXPORT
    extern const char* const kEmptyCollection;
  }

  // Collections internal
  namespace collectionPrinterInternal {

    // Forward declaration, so recursion of containers possible.
    template <typename T>
    CLING_LIB_EXPORT
    std::string printValue(const T* V, const void* = 0);

    template <typename T> inline std::string
    printValue(const T& V, typename std::enable_if<
                             std::is_pointer<decltype(&V)>::value>::type* = 0) {
      return printValue(&V);
    }

    template <typename T0, typename T1> inline std::string
    printValue(const std::pair<T1, T0>* V, const void* AsMap = 0) {
      if (AsMap)
        return printValue(&V->first) + " => " + printValue(&V->second);
      return "{" + printValue(&V->first) + " , " + printValue(&V->second) + "}";
    }

    // For std::vector<bool> elements
    inline std::string printValue(const bool& B, const void* = 0) {
      return cling::printValue(&B);
    }

    struct TypeTest {
      template <class T> static constexpr const void*
      isMap(const T* M, const typename T::mapped_type* V = 0) { return M; }
      static constexpr const void* isMap(const void* M) { return nullptr; }
    };

    // vector, set, deque etc.
    template <typename CollectionType>
    inline auto printValue_impl(
        const CollectionType* obj,
        typename std::enable_if<
            std::is_reference<decltype(*std::begin(*obj))>::value>::type* = 0)
        -> decltype(std::end(*obj), std::string()) {
      auto iter = obj->begin(), iterEnd = obj->end();
      if (iter == iterEnd) return valuePrinterInternal::kEmptyCollection;

      const void* M = TypeTest::isMap(obj);

      std::string str("{ ");
      str += printValue(&(*iter), M);
      while (++iter != iterEnd) {
        str += ", ";
        str += printValue(&(*iter), M);
      }
      return str + " }";
    }

    // As above, but without ability to take address of elements.
    template <typename CollectionType>
    inline auto printValue_impl(
        const CollectionType* obj,
        typename std::enable_if<
            !std::is_reference<decltype(*(obj->begin()))>::value>::type* = 0)
        -> decltype(++(obj->begin()), obj->end(), std::string()) {
      auto iter = obj->begin(), iterEnd = obj->end();
      if (iter == iterEnd) return valuePrinterInternal::kEmptyCollection;

      std::string str("{ ");
      str += printValue(*iter);
      while (++iter != iterEnd) {
        str += ", ";
        str += printValue(*iter);
      }
      return str + " }";
    }
  }

  // Collections
  template<typename CollectionType>
  inline auto printValue(const CollectionType *obj)
  -> decltype(collectionPrinterInternal::printValue_impl(obj), std::string()) {
    return collectionPrinterInternal::printValue_impl(obj);
  }

#if __cplusplus >= 201703L
  // For std::filesystem::path
  inline std::string printValue(const std::filesystem::path* obj) {
    return obj->string();
  }
#endif

  // Arrays
  template<typename T, size_t N>
  inline std::string printValue(const T (*obj)[N]) {
    if (N == 0)
      return valuePrinterInternal::kEmptyCollection;

    std::string str = "{ ";
    str += printValue(*obj + 0);
    for (size_t i = 1; i < N; ++i) {
      str += ", ";
      str += printValue(*obj + i);
    }
    return str + " }";
  }

  // Tuples
  template <class... ARGS>
  inline std::string printValue(std::tuple<ARGS...> *);

  namespace collectionPrinterInternal {
    // We loop at compile time from element 0 to element TUPLE_SIZE - 1
    // of the tuple. The running index is N which has as initial value
    // TUPLE_SIZE. We can therefore stop the iteration and account for the
    // empty tuple case with one single specialisation.
    template <class TUPLE,
              std::size_t N = std::tuple_size<TUPLE>(),
              std::size_t TUPLE_SIZE = std::tuple_size<TUPLE>()>
    struct tuplePrinter {
      static std::string print(TUPLE *t)
      {
        constexpr std::size_t elementNumber = TUPLE_SIZE - N;
        using Element_t = decltype(std::get<elementNumber>(*t));
        std::string ret;
        if (elementNumber)
           ret += ", ";
        ret += cling::printValue(&std::get<elementNumber>(*t));
        // If N+1 is not smaller than the size of the tuple,
        // reroute the call to the printing function to the
        // no-op specialisation to stop recursion.
        constexpr std::size_t Nm1 = N - 1;
        ret += tuplePrinter<TUPLE, Nm1>::print((TUPLE *)t);
        return ret;
      }
    };

    // Special case: no op if last element reached or empty tuple
    template <class TUPLE, std::size_t TUPLE_SIZE>
    struct tuplePrinter<TUPLE, 0, TUPLE_SIZE>
    {
      static std::string print(TUPLE *t) {return "";}
    };

    template <class T>
    inline std::string tuplePairPrintValue(T *val)
    {
      std::string ret("{ ");
      ret += collectionPrinterInternal::tuplePrinter<T>::print(val);
      ret += " }";
      return ret;
    }
  }

  template <class... ARGS>
  inline std::string printValue(std::tuple<ARGS...> *val)
  {
    using T = std::tuple<ARGS...>;
    if (std::tuple_size<T>::value == 0)
      return valuePrinterInternal::kEmptyCollection;
    return collectionPrinterInternal::tuplePairPrintValue<T>(val);
  }

  template <class... ARGS>
  inline std::string printValue(std::pair<ARGS...> *val)
  {
    using T = std::pair<ARGS...>;
    return collectionPrinterInternal::tuplePairPrintValue<T>(val);
  }

  namespace collectionPrinterInternal {
    // Keep this last to allow picking up all specializations above.
    template <typename T> std::string printValue(const T* V, const void*) {
      return cling::printValue(V);
    }
  }

  // unique_ptr<T>:
  template <class T>
  inline std::string printValue(std::unique_ptr<T> *val)
  {
     auto ptr = val->get();
     // printValue dereference its argument. use cast to 'const void**' to get
     // the same printout as a regular pointer.
     return "std::unique_ptr -> " + printValue((const void**)&ptr);
  }

  // shared_ptr<T>:
  template <class T>
  inline std::string printValue(std::shared_ptr<T> *val)
  {
     auto ptr = val->get();
     // printValue dereference its argument. use cast to 'const void**' to get
     // the same printout as a regular pointer.
     return "std::shared_ptr -> " + printValue((const void**)&ptr);
  }

  // weak_ptr<T>:
  template <class T>
  inline std::string printValue(std::weak_ptr<T> *val)
  {
     auto ptr = val->lock().get();
     // printValue dereference its argument. use cast to 'const void**' to get
     // the same printout as a regular pointer.
     return "std::weak_ptr -> " + printValue((const void**)&ptr);
  }

}

#endif
