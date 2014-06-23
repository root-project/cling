namespace M {
  namespace N {
    template<typename T>
    T function(T t) {
      return t;  
    }
    class A{};
    template<typename T>class B : public A {};
    class C :public B<int> {};
    
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