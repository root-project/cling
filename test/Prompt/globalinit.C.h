extern "C" int printf(const char*,...);

struct A {
   struct S {
      S() { printf("A::S()\n"); }
      ~S() { printf("A::~S()\n"); }
   };
   static S s;
};
A::S A::s;
