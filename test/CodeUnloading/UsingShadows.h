#ifndef TEST3

namespace test {
  using ::long_t;
}

namespace test {
  using ::long_t;
}

namespace test {
  using ::long_t;
}

#else

extern "C" {
  double adblf(double);
}
namespace test {
  using ::adblf;

  constexpr float
  adblf(float __x);

  constexpr long double
  adblf(long double __x);
}

using test::adblf;

#endif