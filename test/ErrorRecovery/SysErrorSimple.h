#ifndef CLING_SYSERRR_SIMPLE_2

template <class T>
struct Base {
  unsigned operator()(const T& V) const { return V; }
};

template <class T>
struct Impl {
  unsigned operator()(const T& V) const { return 0; }
};

template<> struct Impl<int> : public Base<int> {
};

#else

class FwdDecl {
};

template<> struct Impl<FwdDecl> {
  unsigned operator()(const FwdDecl& EC) const {
    return Impl<int>()(101);
  }
};

#endif
