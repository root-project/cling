//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// Test runing a file in the same directory `cling CurrentDir.C`
// More info in CIFactory.cpp createCIImpl (line ~850)

// RUN: cd %S && %cling -Xclang -verify CurrentDir.C | FileCheck %s
// Test CurrentDir

extern "C" {
  int printf(const char*, ...);
  char* getcwd(char *buf, std::size_t size);
}

void CurrentDir() {
  char thisDir[1024];
  getcwd(thisDir, sizeof(thisDir));
  /*
    This would be nice, but doesn't work
    printf("We are here: %s\n", thisDir);
    -CHECK: We are here: %S
  */

  printf("Script ran\n");
  // CHECK: Script ran
}

//expected-no-diagnostics
