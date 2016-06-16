
#include <stdlib.h>
extern "C" int printf(const char*,...);

class Test {
public:
  void* operator new  ( size_t count ) {
    printf("Test::operator new\n");
    return ::malloc(count);
  }
  void operator delete(void* ptr) {
    printf("Test::operator delete\n");
    return ::free(ptr);
  }
};