#ifndef TESTB
  #define TEST "TEST 0"
  #define TEST "TEST 1"
  #define TEST "TEST 2"
  #define TEST "TEST 3"
  #define TEST "TEST 4"
#else
  #define TEST "TEST A"
  #undef TEST
  #define TEST "TEST B"
  #undef TEST
  #define TEST "TEST C"
  #undef TEST
  #define TEST "TEST D"
  #undef TEST
  #define TEST "TEST E"
  #undef TEST
  #define TEST "TEST F"
  #define TEST "TEST G"
#endif