
// NonTemplateParmDecls should only print one default fwd decl, i.e it should
// omit the inheritant default arguments.
template<typename, unsigned = 0>
struct __attribute__((type_visibility("default"))) extent;
template<typename, unsigned _Uint>
struct extent{ };

// The same holds for TemplateParmDecls.
template <typename T=int>
class Foo {};
template <typename T>
class Foo;

template <typename T> class DefaultInFwd {};
template <typename T = int> class WithDefaultAndFwd {};
template <typename T = int> class TemplateWithUserForward{};

#include "Def2sub.h"

namespace M {
  namespace N {
    template<typename T>
    T function(T t) {
      return t;
    }
    class A {
      public:
        A(int i = 0) {};
    };
    template<typename T>class B : public A {};
    class C :public B<int> {};

    class D {
      public:
        D(int i = 0) {};
    };

  }
  void FunctionWithDefaultArg(int x=0) {
  }
}
namespace stdtest {
  class istream{};
  extern istream cin;

  template<typename T,typename A> class vector{};
  template<typename... T>class  tuple{};

  template<bool B, class T, class F>
  struct conditional { typedef T type; };

  template<class T, class F>
  struct conditional<false, T, F> { typedef F type; };

  template<bool B, class T = void>
  struct enable_if {};

  template<class T>
  struct enable_if<true, T> { typedef T type; };

}
