/*------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//----------------------------------------------------------------------------*/

// RUN: %cling %s | FileCheck %s

// Ensure the .c file to be processed
// is in the working directory (issue ROOT-6244)
// RUN: cd `dirname %s` ; %cling %s | FileCheck %s


#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

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
