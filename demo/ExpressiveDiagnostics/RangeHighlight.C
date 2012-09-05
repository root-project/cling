// The demo shows when there is an error in a subexpression the relevant 
// subexpression is highlighted
// Author: Vassil Vassilev <vvasilev@cern.ch>

#include <stdio.h>

struct A {
  int X;
};

int func(int val) {
  printf("%d\n",val);
  return val - 10;
}

int RangeHighlight() {
  A SomeA;
  int y = SomeA.X;
  return y + func(y ? ((SomeA.X + 40) + SomeA) / 42 + SomeA.X : SomeA.X);
}
