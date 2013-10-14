// RUN: cat %s | %cling -Xclang -verify

// We must be able to handle cases where, there is a custom function that has 
// attributes non-null arguments and we should be able to add a non-null arg 
// attribute to a say library function.

// Qualified functions.
extern "C" int printf(const char* fmt, ...);
namespace custom_namespace {
  void standaloneFunc(void* p, int q, float* s) __attribute__((nonnull(1,3))) { // expected-warning {{GCC does not allow nonnull attribute in this position on a function definition}}
    if (!p || !s) 
      printf("Must not be called with 0 args.\n");
  }
  void standaloneFunc2(void* p, int q, float* s) __attribute__((nonnull(3)));
  void standaloneFunc2(void* p, int q, float* s) {
    if (!s)
      printf("Must not be called with 0 args.\n");
  }

}
// This can be a function defined in a library or somewhere else. Use printf for example
extern "C" int printf(const char* fmt, ...) __attribute__((nonnull(1)));

int* pNull = 0;
float* fNull = 0;
int* p = new int(1);
float* f = new float(0.0);
const char* charNull = 0;

custom_namespace::standaloneFunc(pNull, 1, fNull); // expected-warning {{null passed to a callee which requires a non-null argument}}
custom_namespace::standaloneFunc(pNull, 1, f); // expected-warning {{null passed to a callee which requires a non-null argument}}
custom_namespace::standaloneFunc(p, 1, fNull); // expected-warning {{null passed to a callee which requires a non-null argument}}
printf(charNull, ""); // expected-warning {{null passed to a callee which requires a non-null argument}}

.rawInput 1
int trampoline() {
  custom_namespace::standaloneFunc(pNull, 1, fNull);
  custom_namespace::standaloneFunc(pNull, 1, f);
  custom_namespace::standaloneFunc(p, 1, fNull);
  return 1;
}
.rawInput 0

//trampoline()

.q
