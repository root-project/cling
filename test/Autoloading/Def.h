int id(int x) {
  return x;
}

class C {
  //irrelevant
};

namespace N {
  void nested() {
  }
}//end namespace N

template<typename T> class Gen {
};

template<> class Gen<int> {
};

template<typename T,typename U> class Partial {
};
