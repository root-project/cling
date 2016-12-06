//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// Test running a file in the same directory `cling CurrentDir.C`.
// More info in CIFactory.cpp ("<<< cling interactive line includer >>>")

// RUN: cp "%s" "%T/CurrentDir.C" && cd %T && %cling -Xclang -verify CurrentDir.C 2>&1 | FileCheck %s
// Test testCurrentDir

extern "C" {
  int printf(const char*, ...);
  char* getcwd(char *buf, std::size_t size);
}

void CurrentDir() {
  printf("Script ran\n"); // CHECK: Script ran
}
//expected-no-diagnostics
