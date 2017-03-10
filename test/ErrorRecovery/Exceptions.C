//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %built_cling -Xclang -verify 2>&1 | FileCheck %s

// Still failing on Windows:
//  typeid() fails when cling was compiled without LLVM_REQUIRES_RTTI / -frtti
//  Heap corruption, possibly due to catch block destruction of exception
// REQUIRES: not_system-windows

#include <cstdint>
#include <stdexcept>
#include <string>
#include <cling/Utils/Platform.h>
#include <cling/Interpreter/Exception.h>

namespace cling {
  namespace internal {
    void TestExceptions(intptr_t Throw);
  }
}

#define TYPE_NAME(X) cling::platform::Demangle(typeid(X).name()).c_str()

#define CLING_TEST_EXCEPT(I, E, F, V) \
  try { cling::internal::TestExceptions(I); } \
  catch(E e) { printf("Caught: %s " F "\n", TYPE_NAME(e), V); } \
  try { cling::internal::TestExceptions(I); } \
  catch(const E& e) { printf("Caught Ref: %s " F "\n", TYPE_NAME(e), V); } \
  try { cling::internal::TestExceptions(-1); } \
  catch(const E& e) { printf("SHOULDN't CATCH\n"); } \
  catch(...) { printf("Caught All Handler\n"); }

#define CLING_TEST_STD_EXCEPT(I, E, F, V) \
  CLING_TEST_EXCEPT(I, E, F, V) \
  /* std::exception inheritance */ \
  try { cling::internal::TestExceptions(I); } \
  catch(std::exception e) { printf("Caught std::exception: %s " F "\n", TYPE_NAME(e), V); } \
  try { cling::internal::TestExceptions(I); } \
  catch(const std::exception& e) { printf("Caught Ref std::exception %s " F "\n", TYPE_NAME(e), V); }

CLING_TEST_STD_EXCEPT(1, std::exception, "%s", e.what())
// CHECK: Caught: std::exception {{.*}}
// CHECK-NEXT: Caught Ref: std::exception {{.*}}
// CHECK-NEXT: Caught All Handler
// CHECK-NEXT: Caught std::exception: std::exception {{.*}}
// CHECK-NEXT: Caught Ref std::exception std::exception {{.*}}

CLING_TEST_STD_EXCEPT(2, std::logic_error, "%s", e.what())
// CHECK-NEXT: Caught: std::logic_error std::logic_error
// CHECK-NEXT: Caught Ref: std::logic_error std::logic_error
// CHECK-NEXT: Caught All Handler
// CHECK-NEXT: Caught std::exception: std::exception {{.*}}
// CHECK-NEXT: Caught Ref std::exception std::logic_error std::logic_error

CLING_TEST_STD_EXCEPT(3, std::runtime_error, "%s", e.what())
// CHECK-NEXT: Caught: std::runtime_error std::runtime_error
// CHECK-NEXT: Caught Ref: std::runtime_error std::runtime_error
// CHECK-NEXT: Caught All Handler
// CHECK-NEXT: Caught std::exception: std::exception {{.*}}
// CHECK-NEXT: Caught Ref std::exception std::runtime_error std::runtime_error

CLING_TEST_STD_EXCEPT(4, std::out_of_range, "%s", e.what())
// CHECK-NEXT: Caught: std::out_of_range std::out_of_range
// CHECK-NEXT: Caught Ref: std::out_of_range std::out_of_range
// CHECK-NEXT: Caught All Handler
// CHECK-NEXT: Caught std::exception: std::exception {{.*}}
// CHECK-NEXT: Caught Ref std::exception std::out_of_range std::out_of_range

CLING_TEST_STD_EXCEPT(5, std::bad_alloc, "%s", e.what())
// CHECK-NEXT: Caught: std::bad_alloc std::bad_alloc
// CHECK-NEXT: Caught Ref: std::bad_alloc std::bad_alloc
// CHECK-NEXT: Caught All Handler
// CHECK-NEXT: Caught std::exception: std::exception {{.*}}
// CHECK-NEXT: Caught Ref std::exception std::bad_alloc std::bad_alloc

CLING_TEST_EXCEPT(6, const char*, "%s", e)
// expected-warning@2 {{duplicate 'const' declaration specifier}}
// expected-warning@2 {{duplicate 'const' declaration specifier}}
// CHECK-NEXT: Caught: char const* c-string
// CHECK-NEXT: Caught Ref: char const* c-string
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT(7, std::string, "%s", e.c_str())
// CHECK-NEXT: Caught: {{.*}} std::string
// CHECK-NEXT: Caught Ref: {{.*}} std::string
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT(10, bool,  "%s", (e ? "true" : "false"))
// CHECK-NEXT: Caught: bool true
// CHECK-NEXT: Caught Ref: bool true
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT(11, float,  "%.8f", e)
// CHECK-NEXT: Caught: float 1.41421354
// CHECK-NEXT: Caught Ref: float 1.41421354
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT(12, double, "%.8f", e)
// CHECK-NEXT: Caught: double 3.14159265
// CHECK-NEXT: Caught Ref: double 3.14159265
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT(-8, int8_t,  "%d", int(e))
// CHECK-NEXT: Caught: signed char -8
// CHECK-NEXT: Caught Ref: signed char -8
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT( 8, uint8_t, "%d", int(e))
// CHECK-NEXT: Caught: unsigned char 8
// CHECK-NEXT: Caught Ref: unsigned char 8
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT(-16, int16_t,  "%d", int(e))
// CHECK-NEXT: Caught: short -16
// CHECK-NEXT: Caught Ref: short -16
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT( 16, uint16_t, "%d", int(e))
// CHECK-NEXT: Caught: unsigned short 16
// CHECK-NEXT: Caught Ref: unsigned short 16
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT(-32, int32_t,  "%d", e)
// CHECK-NEXT: Caught: int -32
// CHECK-NEXT: Caught Ref: int -32
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT( 32, uint32_t, "%u", e)
// CHECK-NEXT: Caught: unsigned int 32
// CHECK-NEXT: Caught Ref: unsigned int 32
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT(-64, int64_t,  "%lld", (long long)e)
// CHECK-NEXT: Caught: long{{( long)?}} -64
// CHECK-NEXT: Caught Ref: long{{( long)?}} -64
// CHECK-NEXT: Caught All Handler

CLING_TEST_EXCEPT( 64, uint64_t, "%llu", (unsigned long long)e)
// CHECK-NEXT: Caught: unsigned long{{( long)?}} 64
// CHECK-NEXT: Caught Ref: unsigned long{{( long)?}} 64
// CHECK-NEXT: Caught All Handler

try {
  int *P = 0;
  *P = 10;
} catch (const cling::InvalidDerefException& E) {
  printf("InvalidDerefException: %s\n", E.what());
}
// CHECK-NEXT: InvalidDerefException: Trying to dereference null pointer or trying to call routine taking non-null arguments

try {
  int *P = 0;
  *P = 10;
} catch (const std::exception& E) {
  printf("std::exception: %s\n", E.what());
}
// CHECK-NEXT: std::exception: Trying to dereference null pointer or trying to call routine taking non-null arguments

throw cling::InterpreterException("From JIT A")
// CHECK-NEXT: >>> Caught an interpreter exception: 'From JIT A'

throw std::runtime_error("From JIT B")
// CHECK-NEXT: >>> Caught a std::exception: 'From JIT B'

throw "StringExcept";
// CHECK-NEXT: >>> Caught an unkown exception.

.q
