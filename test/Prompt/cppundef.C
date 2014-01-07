//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: %cling %s | FileCheck %s
#ifdef MYMACRO
# undef MYMACRO
#endif

extern "C" int printf(const char* fmt, ...);

void MYMACRO(void* i) {
  printf("MYMACRO param=%ld\n", (long)i); // CHECK: MYMACRO param=12
}

void cppundef() {
   MYMACRO((void*)12);
}
