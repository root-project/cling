// RUN: cat %s | %cling | FileCheck --check-prefix=CHECKOUT %s
// RUN: cat %s | %cling 2> /tmp/stderr.txt && cat /tmp/stderr.txt | FileCheck --check-prefix=CHECKERR %s
// RUN: cat %s | %cling | cat /tmp/outfile.txt | FileCheck --check-prefix=CHECK-REDIRECTOUT %s
// RUN: cat %s | %cling | cat /tmp/errfile.txt | FileCheck --check-prefix=CHECK-REDIRECTERR %s
// RUN: cat %s | %cling | cat /tmp/bothfile.txt | FileCheck --check-prefix=CHECK-REDIRECTBOTH %s
// RUN: cat %s | %cling | cat /tmp/anotheroutfile.txt | FileCheck --check-prefix=CHECK-REDIRECTANOTHER %s

#include <iostream>


// Test redirect stdout
.> /tmp/outfile.txt
int a = 10
//CHECK-REDIRECTOUT: (int) 10
int b = 10
//CHECK-REDIRECTOUT: (int) 10
int c = 10
//CHECK-REDIRECTOUT: (int) 10

// Test stderr is not redirected as well.
std::cerr << "Into Error\n";
//CHECKERR: Into Error

// Test toggle back to prompt.
.>
int var = 9
//CHECKOUT: (int) 9

// Test append mode.
.>> /tmp/outfile.txt
a = 99
//CHECK-REDIRECTOUT: (int) 99
b = 99
//CHECK-REDIRECTOUT: (int) 99
c = 99
//CHECK-REDIRECTOUT: (int) 99

// Test redirect stderr
.2> /tmp/errfile.txt
std::cerr << "Error redirected.\n"
//CHECK-REDIRECTERR: Error redirected.

// Test stdout is still redirected to the correct file.
var = 10
//CHECK-REDIRECTOUT: (int) 10

// Test toggle only stdout and stderr still redirected.
.>
a = 100
//CHECKOUT: (int) 100
std::cerr << "Error still redirected.\n"
//CHECK-REDIRECTERR: Error still redirected.

// Test toggle stderr back to prompt.
.2>
std::cerr << "Error back to prompt.\n"
//CHECKERR: Error back to prompt.


// Test redirect of both streams.
.&> /tmp/bothfile.txt
a=10
//CHECK-REDIRECTBOTH: (int) 10
b=10
//CHECK-REDIRECTBOTH: (int) 10
c=10
//CHECK-REDIRECTBOTH: (int) 10
std::cerr << "Redirect both out & err.\n"
//CHECK-REDIRECTBOTH: Redirect both out & err.

// Test toggle both back to the prompt.
.&>
var = 100
//CHECKOUT: (int) 100
std::cerr << "Both back to prompt.\n"
//CHECKERR: Both back to prompt.

// Test append mode for both streams.
.&>> /tmp/bothfile.txt
a=9
//CHECK-REDIRECTBOTH: (int) 9
b=9
//CHECK-REDIRECTBOTH: (int) 9
c=9
//CHECK-REDIRECTBOTH: (int) 9
std::cerr << "Append mode for both streams.\n"
//CHECK-REDIRECTBOTH: Append mode for both streams.


// Test toggle only stdout to prompt and stderr to file.
.>
var = 99
//CHECKOUT: (int) 99
std::cerr << "Err is still in &> file.\n"
//CHECK-REDIRECTBOTH: Err is still in &> file.


// Test toggle stderr to the prompt when redirected with &.
.2>
std::cerr << "Err back from &> file.\n"
//CHECKERR: Err back from &> file.

// Test redirect with & and toggle to out file.
.&>> /tmp/bothfile.txt
var = 999
//CHECK-REDIRECTBOTH: (int) 999
.1> /tmp/anotheroutfile.txt
a = 10
//CHECK-REDIRECTANOTHER: (int) 10
b = 10
//CHECK-REDIRECTANOTHER: (int) 10
c = 10
//CHECK-REDIRECTANOTHER: (int) 10

