//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// Test runing a file in the same directory `cling CurrentDir.C`
// More info in CIFactory.cpp createCIImpl (line ~850)

// RUN: cd %S && %cling -Xclang -verify CurrentDir.C 2>&1 | FileCheck %s
// RUN: mkdir %T/Remove && cd %T/Remove && rm -rf %T/Remove && %cling -DTEST_CWDRETURN %s -Xclang -verify 2>&1 | FileCheck --check-prefix CHECK --check-prefix CHECKcwd %s
// Test testCurrentDir

extern "C" {
  int printf(const char*, ...);
  char* getcwd(char *buf, std::size_t size);
}

#ifdef TEST_CWDRETURN
  // Make sure include still works
  #include <string.h>
  #include <vector>
#endif

void CurrentDir() {
 #ifdef TEST_CWDRETURN
    char thisDir[1024];
    const char *rslt = getcwd(thisDir, sizeof(thisDir));
    // Make sure cling reported the error
    // CHECKcwd: Could not get current working directory: {{.*}}

    if (rslt)
      printf("Working directory exists\n");
    // CHECK-NOT: Working directory exists
 #endif

  printf("Script ran\n");
  // CHECK: Script ran
}

//expected-no-diagnostics
