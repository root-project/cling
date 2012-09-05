extern "C" int printf(const char*,...);

struct B {
   struct S {
      S() { printf("B::S()\n"); }
      ~S() { printf("B::~S()\n"); }
   };
   static S s;
};
B::S B::s;
