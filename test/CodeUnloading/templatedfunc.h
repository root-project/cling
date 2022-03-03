template <typename T> T square(T x) {
  // This unused lambda caused a crash if the file is unloaded; see
  // https://github.com/root-project/root/issues/9850
  auto lambda = [](double x) { return x; };
  return x * x;
}

void templatedfunc() {
  printf("%d\n", square(2));
}
