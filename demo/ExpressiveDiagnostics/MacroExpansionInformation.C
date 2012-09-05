// The demo shows if there was an error in a macro, the expansion location
// macro definition is shown
// Author: Vassil Vassilev <vvasilev@cern.ch>

#define MAX(A, B) ((A) > (B) ? (A) : (B))

struct A {
  int Y;
};

void MacroExpansionInformation () {
  int X = 1;
  A* SomeA = new A();
  X = MAX(X, *SomeA);
}
