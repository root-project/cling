
#if TEST_LITE

class TrailingObjectsBase {
protected: // This is causing the error
  struct OverloadToken {};
};

template <typename BaseTy>
struct TrailingObjectsImpl {
};

template <typename BaseTy>
class TrailingObjects : TrailingObjectsImpl<BaseTy> {
  static int
  callNumTrailingObjects(const BaseTy *Obj,
                         TrailingObjectsBase::OverloadToken) {
    return 1;
  }

  template <typename T>
  static int callNumTrailingObjects(const BaseTy *Obj,
                                    TrailingObjectsBase::OverloadToken) {
    return Obj->numTrailingObjects(TrailingObjectsBase::OverloadToken());
  }
};

#else

namespace trailing_objects_internal {
class TrailingObjectsBase {
protected:
  template <typename T> struct OverloadToken {};
};

template <int Align, typename BaseTy, typename TopTrailingObj, typename PrevTy,
          typename... MoreTys>
struct TrailingObjectsImpl {
};
}

template <typename BaseTy, typename... TrailingTys>
class TrailingObjects : private trailing_objects_internal::TrailingObjectsImpl<
                            1,
                            BaseTy, TrailingObjects<BaseTy, TrailingTys...>,
                            BaseTy, TrailingTys...>
{
  template <int A, typename B, typename T, typename P, typename... M>
  friend struct trailing_objects_internal::TrailingObjectsImpl;

  using TrailingObjectsBase = trailing_objects_internal::TrailingObjectsBase;

  static int
  callNumTrailingObjects(const BaseTy *Obj,
                         TrailingObjectsBase::OverloadToken<BaseTy>) {
    return 1;
  }

  template <typename T>
  static int callNumTrailingObjects(const BaseTy *Obj,
                                    TrailingObjectsBase::OverloadToken<T>) {
    return Obj->numTrailingObjects(TrailingObjectsBase::OverloadToken<T>());
  }
};

#endif
