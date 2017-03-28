class TestA {
  friend class TestB;
  void empty() {}
};

class TestB {
  friend class TestA;
  void empty() {}
};
