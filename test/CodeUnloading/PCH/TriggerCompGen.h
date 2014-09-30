#include "Inputs/CompGen.h"

int CompGen::foo() { return 42; }
CompGen CompGen::Make() { return CompGen(); }

extern "C" int printf(const char*, ...);
void TriggerCompGen() {
   CompGen a;
   CompGen b = a;
   printf("I was executed!\n");
}
