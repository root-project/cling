#ifndef INCLUDE_COMPGEN_H
#define INCLUDE_COMPGEN_H
struct CompGen {
  int I;
  inline int InlineFunc();
};

int CompGen::InlineFunc() { return 17; }
#endif //INCLUDE_COMPGEN_H
