//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify | FileCheck %s

// Test whether the TLS data is properly handled

#include <string>
#include <thread>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>

thread_local unsigned int TLSCounter = 1;
std::mutex WriteMutex;

static void WriteValue(const char* Name, unsigned Val) {
  std::lock_guard<std::mutex> Lock(WriteMutex);
  printf("TLSCounter for '%s' : %u\n", Name, Val);
}
 
void TLSIncrement(const char* Name) {
  ++TLSCounter;
  if (const int Incr = atoi(Name))
    TLSCounter += Incr;
  WriteValue(Name, TLSCounter);
}

static const char* Name[] = {
  "A",
  "B",
  "10",
  "5",
  0,
};
for (unsigned i = 0; Name[i]; ++i) {
  std::thread(TLSIncrement, Name[i]).join();
}
WriteValue("main", TLSCounter);


// CHECK: TLSCounter for 'A' : 2
// CHECK: TLSCounter for 'B' : 2
// CHECK: TLSCounter for '10' : 12
// CHECK: TLSCounter for '5' : 7
// CHECK: TLSCounter for 'main' : 1

// expected-no-diagnostics
.q
