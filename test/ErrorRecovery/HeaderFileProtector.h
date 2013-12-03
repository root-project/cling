#ifndef HEADER_FILE_PROTECTOR
#define HEADER_FILE_PROTECTOR
#define NN 5
#undef NN
#define NN 6
#define P 5

class MyHeaderFileProtectedClass {};
#undef P
error_here; // expected-error {{C++ requires a type specifier for all declarations}}
#endif // HEADER_FILE_PROTECTOR
