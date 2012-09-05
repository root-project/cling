// RUN: %cling %s | FileCheck %s

extern "C" int printf(const char*,...);

const char* defaultArgV[] = {"A default argument", "", 0};

int test_01(int argc=12, const char** argv = defaultArgV)
{
  int i;
  for( i = 0; i < 5; ++i )
    printf( "Hello World #%d\n", i );
  // CHECK: Hello World #0
  // CHECK: Hello World #1
  // CHECK: Hello World #2
  // CHECK: Hello World #3
  // CHECK: Hello World #4
  return 0;
}
