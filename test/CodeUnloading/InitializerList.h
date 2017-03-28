
#include <initializer_list>

template <class T, unsigned N>
class TestIList {
  T m_Data[N];
public:
  TestIList(std::initializer_list<T> l) {
    int i = 0;
    for (T val: l )
      m_Data[i++] = val;
  }
  T sum() const {
    T rval = 0;
    for (int i = 0; i < N; ++i)
      rval += m_Data[i];
    return rval;
  }
};
