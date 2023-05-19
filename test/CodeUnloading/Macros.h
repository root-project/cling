#ifndef TESTB
  #define TEST "TEST 0"
  #define TEST "TEST 1"
  #define TEST "TEST 2"
  #define TEST "TEST 3"
  #define TEST "TEST 4"
  // expected-warning@3 {{'TEST' macro redefined}}
  // expected-note@2 {{previous definition is here}}
  // expected-warning@4 {{'TEST' macro redefined}}
  // expected-note@3 {{previous definition is here}}
  // expected-warning@5 {{'TEST' macro redefined}}
  // expected-note@4 {{previous definition is here}}
  // expected-warning@6 {{'TEST' macro redefined}}
  // expected-note@5 {{previous definition is here}}
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
  // expected-warning@27 {{'TEST' macro redefined}}
  // expected-note@26 {{previous definition is here}}
#endif
