
namespace TEST_NAMESPACE {
  inline namespace nested {
    class Nested;
  }
}

namespace TEST_NAMESPACE {
  class Test {
  public:
    Test(const Nested&);
  };
}
