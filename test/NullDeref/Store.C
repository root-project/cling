// RUN: cat %s | %cling | FileCheck %s

//This file checks a pointer store operation for null prt dereference.
int *p;
*p = 6; // CHECK: Warning: you are about to dereference null ptr, which probably will lead to seg violation. Do you want to proceed?[y/n]
n
