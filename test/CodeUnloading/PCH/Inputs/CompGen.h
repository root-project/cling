#ifndef INCLUDE_COMPGEN_H
#define INCLUDE_COMPGEN_H
struct CompGen {
   int I;
   virtual int foo();
   static CompGen Make();
};
#endif // INCLUDE_COMPGEN_H
