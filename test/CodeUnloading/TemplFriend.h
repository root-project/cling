#include <iterator>
#include <vector>

struct Test {
  std::vector<int> Vec;
  typedef std::vector<int>::const_iterator iterator;
  iterator begin() const { return Vec.begin(); }
  iterator end() const { return Vec.end(); }
  Test(int i) : Vec(i, i) {}
};
