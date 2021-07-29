//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling -Xclang -verify 2>&1 | FileCheck %s
// Tests continuation of a line with , or \
// Be careful saving this file: some editors strip the trailing spaces at bottom

extern "C" int printf(const char*, ...);

int Ac = 15,
    Bc = 25,
    Cc = 35;

Ac
// CHECK: (int) 15
Bc
// CHECK-NEXT: (int) 25
Cc
// CHECK-NEXT: (int) 35

// Should not enter line continuation mode here (ROOT-9202) 
unsigned u1 = 45, u2
// CHECK-NEXT: (unsigned int) {{[[:digit:]]+}}
u1
// CHECK-NEXT: (unsigned int) 45
int i1 \ i2 // expected-error {{expected ';' at end of declaration}}

static void InvokeTest(int A,
                       int B) { printf("Invoke: %d, %d\n", A, B); }
InvokeTest(Ac,
 Bc);
// CHECK-NEXT: Invoke: 15, 25

int A = 10,   \
    B = 20,   \
    C = 30;

A
// CHECK-NEXT: (int) 10
B
// CHECK-NEXT: (int) 20
C
// CHECK-NEXT: (int) 30
		
#define CLING_MULTILINE_STRING  "A" \
 "B" \
 " C D"

CLING_MULTILINE_STRING
// CHECK-NEXT: (const char [7]) "AB C D"

"Multinline" \
 " String " \
  "Constant"
"Separate"

// CHECK-NEXT: (const char [27]) "Multinline String Constant"
// CHECK-NEXT: (const char [9]) "Separate"

// Common error handling macro
#define CLING_MULTILINE_MACRO(STR)  do { \
   printf(STR "\n"); } while(0)

CLING_MULTILINE_MACRO("DOWHILE");
// CHECK-NEXT: DOWHILE

#define CLING_MULTILINE_TRAILING_SPACE   \    
  "Trailing Space "   \    
  "And A Tab" \		
  " End" // expected-warning@1 {{backslash and newline separated by space}} // expected-warning@2 {{backslash and newline separated by space}}

CLING_MULTILINE_TRAILING_SPACE
// CHECK-NEXT: (const char [29]) "Trailing Space And A Tab End"

.q
