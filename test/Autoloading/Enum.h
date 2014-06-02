enum class EC {
  A,
  B,
  C
};

enum E : unsigned int {
  E_a,
  E_b,
  E_c
};

template<> class Gen<E> {
};