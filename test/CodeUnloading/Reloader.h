
#if !defined(POSE_AS_STD_NAMESPACE_)
 #if defined(_LIBCPP_VERSION)
  #define POSE_AS_STD_NAMESPACE_    _LIBCPP_BEGIN_NAMESPACE_STD
  #define _POSE_AS_STD_NAMESPACE    _LIBCPP_END_NAMESPACE_STD
 #elif defined(_GLIBCXX_VISIBILITY)
  #define POSE_AS_STD_NAMESPACE_    namespace std _GLIBCXX_VISIBILITY(default) { _GLIBCXX_BEGIN_NAMESPACE_VERSION
  #define _POSE_AS_STD_NAMESPACE    _GLIBCXX_END_NAMESPACE_VERSION }
 #else
  #define POSE_AS_STD_NAMESPACE_    _STD_BEGIN
  #define _POSE_AS_STD_NAMESPACE    _STD_END
 #endif
#endif

POSE_AS_STD_NAMESPACE_

#if defined(POSE_NOT_TEMPLATED)
  class string {
    void* m_Pad[10];
    const char* m_Val;
  public:
    string(const char* CStr) : m_Val(CStr) {}
    const char* c_str() const { return m_Val; }
  };
#else
  template <class T> class char_traits {};
  template <class T> class allocator {};
  template<class Elem, class Traits, class Alloc>
  class basic_string {
    void* m_Pad[20];
    const char* m_Val;
  public:
    basic_string(const char* CStr) : m_Val(CStr) {}
    const char* c_str() const { return m_Val; }
  };

  typedef basic_string<char, char_traits<char>, allocator<char> > string;
#endif

_POSE_AS_STD_NAMESPACE
