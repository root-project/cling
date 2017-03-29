//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s

#include <stdio.h>

struct TERD {
    const char *Name;
    TERD(const char *N) : Name(N) { printf("TERD::TERD::%s\n", Name); }
    ~TERD() { printf("TERD::~TERD::%s\n", Name); }
};
static TERD& inst01() {
    static TERD st01("inst01");
    return st01;
}
static TERD& inst02() {
    static TERD st02("inst02");
    return st02;
}

inst01();
inst01();
inst01();
//      CHECK: TERD::TERD::inst01

inst02();
inst02();
inst02();
// CHECK-NEXT: TERD::TERD::inst02

// expected-no-diagnostics
.q

// CHECK-NEXT: TERD::~TERD::inst02
// CHECK-NEXT: TERD::~TERD::inst01
