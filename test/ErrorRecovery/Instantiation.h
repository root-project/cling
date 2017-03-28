
#ifdef TEST_CLASS

TEST_TMPLT class TEST_CLASS {
  typedef TEST_VALT value_type;
  value_type m_Size;
public:
  class iterator {
    value_type m_Val;
  public:
    iterator(value_type val) : m_Val(val) {}
    bool operator != ( const iterator &rhs ) const { return m_Val != rhs.m_Val; }
    iterator& operator ++ () { ++m_Val; return *this; }
    value_type operator * () const { return m_Val; };
  };
  iterator begin() const { return iterator(0); }
  iterator end()   const { return iterator(m_Size); }
  TEST_CLASS & operator() (value_type val) { m_Size = val; return *this; }
};

#else

template <class T>
class TestIter {
  T m_Val;
public:
  TestIter(T val) : m_Val(val) {}
  bool operator != ( const TestIter<T> &rhs ) const { return m_Val != rhs.m_Val; }
  TestIter& operator ++ () { ++m_Val; return *this; }
  T operator * () const { return m_Val; };
};

template <class T>
class TestIterInst {
  T m_Size;
public:
  bool operator == ( const TestIterInst<T>& rhs ) const { return false; }
  TestIter<T> begin() const { return TestIter<T>(0); }
  TestIter<T> end()   const { return TestIter<T>(m_Size); }
  TestIterInst&    test1(T val) { m_Size = val; return *this; }
  TestIterInst<T>& test2(T val) { m_Size = val; return *this; }
  bool access() const { return testAccessDecl(); }
private:
  bool testAccessDecl() const { return true; }
};

template <class T>
class TestStatic {
  static T s_Counter;
public:
  TestStatic() { ++s_Counter; }
  TestIter<T> begin() const { return TestIter<T>(0); }
  TestIter<T> end()   const { return TestIter<T>(s_Counter); }
};
template<> int TestStatic<int>::s_Counter = 0;

#endif
