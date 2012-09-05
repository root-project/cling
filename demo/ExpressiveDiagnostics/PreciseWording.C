// The demo shows the Fix-it hints that try to guess what the user had meant
// when he did the error
// Author: Vassil Vassilev <vvasilev@cern.ch>

struct A {
  int X;
};

int PreciseWording () {
  A SomeA;
  int y = *SomeA.X;
}
