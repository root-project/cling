#ifndef HEADER_FILE_PROTECTOR
#define HEADER_FILE_PROTECTOR
int f() {
  return NN+1;
}
int n = f();
error_here; // expected-error {{C++ requires a type specifier for all declarations}}
#endif // HEADER_FILE_PROTECTOR
