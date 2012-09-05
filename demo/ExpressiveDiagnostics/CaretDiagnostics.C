// The demo shows that cling (clang) can report even the column number of the 
// error and emit caret
// Author: Vassil Vassilev <vvasilev@cern.ch>

#include <stdio.h>

void CaretDiagnostics() {
  int i = 5;
  printf("%.*d\n",i);
}
