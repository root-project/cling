//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify
// XFAIL: powerpc64

// We must be able to handle cases where, there is a custom function that has
// attributes non-null arguments and we should be able to add a non-null arg
// attribute to a say library function.

// Qualified functions.
extern "C" int printf(const char* fmt, ...);
namespace custom_namespace {
  void standaloneFunc(void* p, int q, float* s) __attribute__((nonnull(1,3))) { // expected-warning {{GCC does not allow 'nonnull' attribute in this position on a function definition}} //expected-note@2{{declared 'nonnull' here}} //expected-note@2{{declared 'nonnull' here}}
    if (!p || !s) // expected-warning {{nonnull parameter 'p' will evaluate to 'true' on first encounter}} // expected-warning {{nonnull parameter 's' will evaluate to 'true' on first encounter}}
      printf("Must not be called with 0 args.\n");
  }
  void standaloneFunc2(void* p, int q, float* s) __attribute__((nonnull(3))); //expected-note@6{{declared 'nonnull' here}}
  void standaloneFunc2(void* p, int q, float* s) {
    if (!s) // expected-warning {{nonnull parameter 's' will evaluate to 'true' on first encounter}}
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

custom_namespace::standaloneFunc(pNull, 1, fNull); // expected-warning {{null passed to a callee that requires a non-null argument}}
custom_namespace::standaloneFunc(pNull, 1, f); // expected-warning {{null passed to a callee that requires a non-null argument}}
custom_namespace::standaloneFunc(p, 1, fNull); // expected-warning {{null passed to a callee that requires a non-null argument}}
printf(charNull, ""); // expected-warning {{null passed to a callee that requires a non-null argument}}

int trampoline() {
  custom_namespace::standaloneFunc(pNull, 1, fNull);
  custom_namespace::standaloneFunc(pNull, 1, f);
  custom_namespace::standaloneFunc(p, 1, fNull);
  return 1;
}

//trampoline()

.q
